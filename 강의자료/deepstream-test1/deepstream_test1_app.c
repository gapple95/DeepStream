

/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr)                \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr)             \
  {                                                       \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1;                                            \
  }

gint frame_number = 0;
gchar pgie_classes_str[4][32] = {"Vehicle", "TwoWheeler", "Person",
                                 "Roadsign"};

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
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
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
    offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

    /* Now set the offsets where the string should appear */
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    /* Font , font-color and font-size */
    txt_params->font_params.font_name = "Serif";
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    /* Text background color */
    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }

  g_print("Frame Number = %d Number of objects = %d "
          "Vehicle Count = %d Person Count = %d\n",
          frame_number, num_rects, vehicle_count, person_count);
  frame_number++;
  return GST_PAD_PROBE_OK;
}

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
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
             *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, 
             *nvvidconv = NULL, *nvosd = NULL, *enc = NULL, *filesink = NULL;

  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  gboolean yaml_config = FALSE;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc != 2)
  {
    g_printerr("Usage: %s <yml file>\n", argv[0]);
    g_printerr("OR: %s <H264 filename>\n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  /* Parse inference plugin type */
  yaml_config = (g_str_has_suffix(argv[1], ".yml") ||
                 g_str_has_suffix(argv[1], ".yaml"));

  if (yaml_config)
  {
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
                                               "primary-gie"));
  }

  /* Create gstreamer elements */
  pipeline = gst_pipeline_new("dstest1-pipeline");

  /* Source element for reading from the file */
  source = gst_element_factory_make("filesrc", "file-source");
  h264parser = gst_element_factory_make("h264parse", "h264-parser");
  decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
  pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

  /* New elements for saving the output */
  enc = gst_element_factory_make("nvv4l2h264enc", "h264-encoder");
  filesink = gst_element_factory_make("filesink", "file-sink");

  /* Check if elements are created successfully */
  if (!pipeline || !source || !h264parser || !decoder || !streammux || 
      !pgie || !nvvidconv || !nvosd || !enc || !filesink)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Set element properties */
  g_object_set(G_OBJECT(source), "location", argv[1], NULL);
  g_object_set(G_OBJECT(streammux), "batch-size", 1, "width", MUXER_OUTPUT_WIDTH,
               "height", MUXER_OUTPUT_HEIGHT, "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  g_object_set(G_OBJECT(pgie), "config-file-path", "dstest1_pgie_config.txt", NULL);
  g_object_set(G_OBJECT(filesink), "location", "output.mp4", NULL);

  /* Add elements to the pipeline */
  gst_bin_add_many(GST_BIN(pipeline),
                   source, h264parser, decoder, streammux, pgie,
                   nvvidconv, nvosd, enc, filesink, NULL);

  /* Link elements together */
  if (!gst_element_link_many(source, h264parser, decoder, NULL))
  {
    g_printerr("Elements could not be linked: source to decoder. Exiting.\n");
    return -1;
  }

  if (!gst_element_link_many(streammux, pgie, nvvidconv, nvosd, enc, filesink, NULL))
  {
    g_printerr("Elements could not be linked: streammux to filesink. Exiting.\n");
    return -1;
  }

  /* Set the pipeline to "playing" state */
  g_print("Using file: %s\n", argv[1]);
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Clean up */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);
  return 0;
}