import sys
import time
import json
import paho.mqtt.client as mqtt
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib

# MQTT 설정
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "deepstream_person_detect"

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# DeepStream 초기화
Gst.init(None)

def send_mqtt_message(obj_meta):
    """MQTT로 감지된 객체 데이터를 송신"""
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
    """DeepStream에서 감지된 객체의 정보를 MQTT로 전송"""
    import pyds
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        print("Unable to get batch_meta")
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                if obj_meta.class_id == 2:  # 'person' 클래스
                    send_mqtt_message(obj_meta)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
        except StopIteration:
            break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def main():
    """DeepStream 파이프라인을 실행"""
    pipeline = Gst.parse_launch(
        "filesrc location=sample_video.mp4 ! decodebin ! nvstreammux name=mux batch-size=1 width=1280 height=720 ! nvinfer config-file-path=dstest1_pgie_config.txt ! nvvideoconvert ! nvdsosd ! nveglglessink"
    )

    osdsink = pipeline.get_by_name("nvosd")
    if osdsink is None:
        print("Unable to get nvosd element")
        sys.exit(1)

    sink_pad = osdsink.get_static_pad("sink")
    if sink_pad is None:
        print("Unable to get sink pad")
        sys.exit(1)

    sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

    print("Starting DeepStream pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Exiting...")
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
