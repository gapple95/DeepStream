/*
 * Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. 
 * Should ideally be set based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr)                \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr)             \
  {                                                       \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1;                                            \
  }

gint frame_number = 0;
gchar pgie_classes_str[4][32] = {"Vehicle", "TwoWheeler", "Person", "Roadsign"};

/* Probe function to parse metadata and print object counts */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  guint num_rects = 0;
  guint vehicle_count = 0;
  guint person_count = 0;

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
      if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
      {
        vehicle_count++;
        num_rects++;
      }
      if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
      {
        person_count++;
        num_rects++;
      }
    }
  }

  g_print("Frame Number = %d Number of objects = %d "
          "Vehicle Count = %d Person Count = %d\n",
          frame_number, num_rects, vehicle_count, person_count);

  frame_number++;
  return GST_PAD_PROBE_OK;
}

/* Handle GstBus messages */
static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;
  case GST_MESSAGE_ERROR:
  {
    gchar *debug = NULL;
    GError *error = NULL;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }
  default:
    break;
  }
  return TRUE;
}

int main(int argc, char *argv[])
{
  GMainLoop *loop = NULL;

  /* GStreamer elements */
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
             *decoder = NULL, *streammux = NULL, *pgie = NULL,
             *nvvidconv = NULL, *nvosd = NULL;

  /* For MP4 output */
  GstElement *enc = NULL, *h264parser_out = NULL, *mux = NULL, *filesink = NULL;

  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;

  gboolean yaml_config = FALSE;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;

  /* CUDA device info */
  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc != 2)
  {
    g_printerr("Usage: %s <H264 file or .yml/.yaml>\n", argv[0]);
    return -1;
  }

  /* Initialize GStreamer */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  /* Check if we're using a YML config */
  yaml_config = (g_str_has_suffix(argv[1], ".yml") ||
                 g_str_has_suffix(argv[1], ".yaml"));

  if (yaml_config)
  {
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie"));
  }

  /* Create gstreamer elements */
  pipeline = gst_pipeline_new("dstest1-pipeline");

  /* Source: reading from file (H264 elementary stream) */
  source = gst_element_factory_make("filesrc", "file-source");
  h264parser = gst_element_factory_make("h264parse", "h264-parser");
  decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");

  /* streammux: batch multiple streams (here we use only 1) */
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  /* Primary inference (PGIE) */
  /* (If nvinferserver is needed, check pgie_type) */
  pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");

  /* Convert NV12->RGBA for OSD */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

  /* OSD to draw bounding boxes etc. */
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

  /* For MP4 output: H.264 encoder -> parser -> MP4 mux -> filesink */
  enc = gst_element_factory_make("nvv4l2h264enc", "h264-encoder");
  h264parser_out = gst_element_factory_make("h264parse", "h264-parser-out");
  mux = gst_element_factory_make("qtmux", "qtmux");
  filesink = gst_element_factory_make("filesink", "file-sink");

  if (!pipeline || !source || !h264parser || !decoder || !streammux ||
      !pgie || !nvvidconv || !nvosd || !enc || !h264parser_out || !mux || !filesink)
  {
    g_printerr("Failed to create one or more GStreamer elements.\n");
    return -1;
  }

  /* Set element properties */
  g_object_set(G_OBJECT(source), "location", argv[1], NULL);

  /* streammux properties */
  g_object_set(G_OBJECT(streammux), "batch-size", 1, 
               "width", MUXER_OUTPUT_WIDTH,
               "height", MUXER_OUTPUT_HEIGHT,
               "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, 
               NULL);

  /* pgie config file (if needed) */
  if (!yaml_config)
  {
    /* 예시: txt config 사용 */
    g_object_set(G_OBJECT(pgie), 
      "config-file-path", "dstest1_pgie_config.txt", 
      NULL);
  }
  else
  {
    /* YAML config 파싱 */
    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux, argv[1], "streammux"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie, argv[1], "primary-gie"));
  }

  /* filesink for final output */
  g_object_set(G_OBJECT(filesink), 
               "location", "output.mp4", 
               /* 파일 저장 시에는 sync, async를 끄는 경우 많음 */
               "sync", FALSE,
               "async", FALSE,
               NULL);

  /* Optional: set bitrate on encoder, e.g. 4Mbps */
  g_object_set(G_OBJECT(enc), "bitrate", 4000000, NULL);

  /* Add elements to pipeline */
  gst_bin_add_many(GST_BIN(pipeline),
                   source, h264parser, decoder,
                   streammux, pgie, nvvidconv, nvosd,
                   enc, h264parser_out, mux, filesink,
                   NULL);

  /* Bus and watch */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  /* ---- Link elements ---- */

  /* 1) source -> h264parser -> decoder */
  if (!gst_element_link_many(source, h264parser, decoder, NULL))
  {
    g_printerr("Could not link source->parser->decoder\n");
    return -1;
  }

  /* 2) decoder src pad -> streammux sink pad */
  {
    GstPad *decoder_src_pad = gst_element_get_static_pad(decoder, "src");
    GstPad *streammux_sink_pad = gst_element_request_pad_simple(streammux, "sink_0");
    if (!decoder_src_pad || !streammux_sink_pad)
    {
      g_printerr("Failed to get pads from decoder/streammux\n");
      return -1;
    }
    if (gst_pad_link(decoder_src_pad, streammux_sink_pad) != GST_PAD_LINK_OK)
    {
      g_printerr("Failed to link decoder src to streammux sink\n");
      gst_object_unref(decoder_src_pad);
      gst_object_unref(streammux_sink_pad);
      return -1;
    }
    gst_object_unref(decoder_src_pad);
    gst_object_unref(streammux_sink_pad);
  }

  /* 3) streammux -> pgie -> nvvidconv -> nvosd */
  if (!gst_element_link_many(streammux, pgie, nvvidconv, nvosd, NULL))
  {
    g_printerr("Could not link streammux->pgie->nvvidconv->nvosd\n");
    return -1;
  }

  /* OSD에서 메타 정보 확인 (probe) */
  osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
  if (!osd_sink_pad)
  {
    g_printerr("Unable to get sink pad from nvosd\n");
  }
  else
  {
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref(osd_sink_pad);
  }

  /* 4) nvosd -> encoder -> h264parse_out -> qtmux -> filesink */
  if (!gst_element_link_many(nvosd, enc, h264parser_out, mux, filesink, NULL))
  {
    g_printerr("Could not link nvosd->enc->parser_out->mux->filesink\n");
    return -1;
  }

  /* Set pipeline to playing */
  g_print("Using file: %s\n", argv[1]);
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Cleanup */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);

  return 0;
}
