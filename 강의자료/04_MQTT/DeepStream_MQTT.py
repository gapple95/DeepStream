import gi
import sys
import json
import paho.mqtt.client as mqtt
from gi.repository import Gst, GLib

gi.require_version('Gst', '1.0')
Gst.init(None)

# MQTT 설정
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "deepstream_person_detect"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

def send_mqtt_message(obj_meta):
    """감지된 객체 정보를 MQTT로 송신"""
    payload = {
        "object": "person",
        "confidence": obj_meta.confidence,
        "position": {
            "left": obj_meta.rect_params.left,
            "top": obj_meta.rect_params.top,
            "width": obj_meta.rect_params.width,
            "height": obj_meta.rect_params.height
        }
    }
    client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
    print(f"Sent MQTT message: {payload}")

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """DeepStream에서 감지된 객체가 'person'이면 MQTT로 전송"""
    import pyds
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK
    
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            if obj_meta.class_id == 2:
                send_mqtt_message(obj_meta)
            l_obj = l_obj.next
        l_frame = l_frame.next
    
    return Gst.PadProbeReturn.OK

def create_pipeline():
    """DeepStream GStreamer 파이프라인 생성"""
    pipeline = Gst.Pipeline.new("deepstream-pipeline")
    
    source = Gst.ElementFactory.make("v4l2src", "camera-source")
    caps = Gst.ElementFactory.make("capsfilter", "source-caps")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw,framerate=30/1"))
    
    vidconv_src = Gst.ElementFactory.make("videoconvert", "source-convert")
    nvvidconv_src = Gst.ElementFactory.make("nvvidconv", "nvidia-convert")
    caps_nvmm = Gst.ElementFactory.make("capsfilter", "nvmm-caps")
    caps_nvmm.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 40000)
    
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", "./dstest1_pgie_config.txt")
    
    nvvidconv = Gst.ElementFactory.make("nvvidconv", "video-converter")
    nvosd = Gst.ElementFactory.make("nvdsosd", "on-screen-display")
    sink = Gst.ElementFactory.make("nveglglessink", "display")
    
    elements = [
        source, caps, vidconv_src, nvvidconv_src, caps_nvmm,
        streammux, pgie, nvvidconv, nvosd, sink
    ]
    
    for element in elements:
        if not element:
            sys.stderr.write(f"{element} 생성 실패\n")
            sys.exit(1)
        pipeline.add(element)
    
    source.link(caps)
    caps.link(vidconv_src)
    vidconv_src.link(nvvidconv_src)
    nvvidconv_src.link(caps_nvmm)
    
    srcpad = caps_nvmm.get_static_pad("src")
    sinkpad = streammux.request_pad_simple("sink_0")
    srcpad.link(sinkpad)
    
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)
    
    return pipeline

pipeline = create_pipeline()
loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()

def on_message(bus, message):
    if message.type == Gst.MessageType.EOS:
        print("스트림이 종료되었습니다.")
        loop.quit()
    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"오류: {err}, 디버그 정보: {debug}")
        loop.quit()

bus.connect("message", on_message)

nvosd_element = pipeline.get_by_name("on-screen-display")
if not nvosd_element:
    print("오류: 'nvdsosd' 요소를 찾을 수 없습니다.")
    sys.exit(1)

sink_pad = nvosd_element.get_static_pad("sink")
sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

pipeline.set_state(Gst.State.PLAYING)
try:
    print("파이프라인 실행 중...")
    loop.run()
except KeyboardInterrupt:
    print("종료 중...")
finally:
    pipeline.set_state(Gst.State.NULL)
