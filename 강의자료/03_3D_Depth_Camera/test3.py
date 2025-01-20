import depthai as dai
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# GStreamer 초기화
Gst.init(None)

# GStreamer 파이프라인 생성 함수
def create_gstreamer_pipeline():
    pipeline_str = """
        appsrc name=source ! videoconvert ! video/x-raw,format=BGR,width=1920,height=1080 ! \
        nvvideoconvert gpu-id=0 ! video/x-raw(memory:NVMM),format=NV12,width=1920,height=1080 ! \
        nvstreammux name=muxer width=1920 height=1080 batch-size=1 batched-push-timeout=40000 ! \
        nvinfer config-file-path=./dstest1_pgie_config.txt ! \
        nvdsosd gpu-id=0 ! \
        nvvidconv gpu-id=0 ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink
    """
    return Gst.parse_launch(pipeline_str)

# OAK-D 파이프라인 생성 함수
def create_oak_d_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(1920, 1080)  # 해상도 동기화
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("video")
    cam.preview.link(xout.input)

    return pipeline

# 메타데이터 디버깅 함수
def process_metadata(sample):
    """GStreamer 추론 결과 메타데이터 디버깅"""
    buf = sample.get_buffer()
    print("Metadata buffer received:", buf)

# GStreamer에서 처리된 프레임 가져오기
def get_frame_from_gstreamer(sink):
    sample = sink.emit("pull-sample")
    if not sample:
        return None

    process_metadata(sample)
    return True

# GStreamer로 프레임 전달 함수
def push_frame_to_gstreamer(source, frame):
    height, width, _ = frame.shape
    buf = Gst.Buffer.new_allocate(None, frame.nbytes, None)
    buf.fill(0, frame.tobytes())
    buf.pts = Gst.util_uint64_scale(Gst.CLOCK_TIME_NONE, 1, Gst.SECOND)

    caps = Gst.Caps.from_string(f"video/x-raw,format=BGR,width={width},height={height},framerate=30/1")
    source.set_property("caps", caps)
    source.emit("push-buffer", buf)

# 메인 실행
def main():
    gst_pipeline = None
    try:
        gst_pipeline = create_gstreamer_pipeline()
        source = gst_pipeline.get_by_name("source")
        sink = gst_pipeline.get_by_name("sink")
        gst_pipeline.set_state(Gst.State.PLAYING)

        oak_d_pipeline = create_oak_d_pipeline()
        with dai.Device(oak_d_pipeline) as device:
            queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

            while True:
                in_frame = queue.get()
                frame = in_frame.getCvFrame()

                push_frame_to_gstreamer(source, frame)
                if get_frame_from_gstreamer(sink):
                    print("Inference completed.")

    except Exception as e:
        print("Error:", e)
    finally:
        if gst_pipeline:
            gst_pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped.")

if __name__ == "__main__":
    main()
