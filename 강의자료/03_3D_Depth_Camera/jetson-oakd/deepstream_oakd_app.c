#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "depthai/depthai.hpp"

// GStreamer appsrc와 연결된 OAK-D 데이터를 처리하는 쓰레드
static gboolean push_data_to_appsrc(GstElement *appsrc)
{
  static auto device = std::make_unique<dai::Device>();
  static auto videoQueue = device->getOutputQueue("video", 30, false);

  auto frame = videoQueue->get<dai::ImgFrame>();
  if (!frame)
    return TRUE;

  GstBuffer *buffer = gst_buffer_new_allocate(NULL, frame->getData().size(), NULL);
  gst_buffer_fill(buffer, 0, frame->getData().data(), frame->getData().size());
  GstFlowReturn ret = gst_app_src_push_buffer((GstAppSrc *)appsrc, buffer);

  if (ret != GST_FLOW_OK)
  {
    g_printerr("Failed to push buffer to appsrc\n");
    return FALSE; // Stop pushing data
  }
  return TRUE;
}

int main(int argc, char *argv[])
{
  GstElement *pipeline, *appsrc, *streammux, *pgie, *nvvidconv, *nvosd, *sink;
  GMainLoop *loop;

  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  // 파이프라인 생성
  pipeline = gst_pipeline_new("dsoakd-pipeline");
  appsrc = gst_element_factory_make("appsrc", "camera-source");
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
  pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
  sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");

  if (!pipeline || !appsrc || !streammux || !pgie || !nvvidconv || !nvosd || !sink)
  {
    g_printerr("Failed to create GStreamer elements\n");
    return -1;
  }

  // appsrc 설정
  g_object_set(G_OBJECT(appsrc),
               "caps", gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGB", "width", G_TYPE_INT, 640, "height", G_TYPE_INT, 480, "framerate", GST_TYPE_FRACTION, 30, 1, NULL),
               NULL);

  // streammux 설정
  g_object_set(G_OBJECT(streammux), "batch-size", 1, NULL);

  // pgie 설정
  g_object_set(G_OBJECT(pgie), "config-file-path", "dsoakd_pgie_config.txt", NULL);

  // 파이프라인 연결
  gst_bin_add_many(GST_BIN(pipeline), appsrc, streammux, pgie, nvvidconv, nvosd, sink, NULL);
  if (!gst_element_link_many(streammux, pgie, nvvidconv, nvosd, sink, NULL))
  {
    g_printerr("Failed to link elements\n");
    return -1;
  }

  // appsrc -> streammux 연결
  GstPad *streammux_sink_pad = gst_element_get_request_pad(streammux, "sink_0");
  GstPad *appsrc_src_pad = gst_element_get_static_pad(appsrc, "src");
  if (gst_pad_link(appsrc_src_pad, streammux_sink_pad) != GST_PAD_LINK_OK)
  {
    g_printerr("Failed to link appsrc to streammux\n");
    return -1;
  }
  gst_object_unref(streammux_sink_pad);

  // appsrc에서 데이터 푸시
  g_timeout_add(30, (GSourceFunc)push_data_to_appsrc, appsrc);

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}
