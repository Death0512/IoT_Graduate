/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "nvdsinfer_custom_impl.h"

#include "utils.h"

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

static NvDsInferParseObjectInfo
convertBBox(const float& bx1, const float& by1, const float& bx2, const float& by2, const uint& netW, const uint& netH)
{
  NvDsInferParseObjectInfo b;

  float x1 = bx1;
  float y1 = by1;
  float x2 = bx2;
  float y2 = by2;

  x1 = clamp(x1, 0, netW);
  y1 = clamp(y1, 0, netH);
  x2 = clamp(x2, 0, netW);
  y2 = clamp(y2, 0, netH);

  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  return b;
}

static void
addBBoxProposal(const float bx1, const float by1, const float bx2, const float by2, const uint& netW, const uint& netH,
    const int maxIndex, const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
  NvDsInferParseObjectInfo bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);

  if (bbi.width < 1 || bbi.height < 1) {
    return;
  }

  bbi.detectionConfidence = maxProb;
  bbi.classId = maxIndex;
  binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYolo(const float* output, const uint& outputSize, const uint& netW, const uint& netH,
    const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = output[b * 6 + 4];
    int maxIndex = (int) output[b * 6 + 5];

    if (maxProb < preclusterThreshold[maxIndex]) {
      continue;
    }

    float bx1 = output[b * 6 + 0];
    float by1 = output[b * 6 + 1];
    float bx2 = output[b * 6 + 2];
    float by2 = output[b * 6 + 3];

    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
  }

  return binfo;
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYolo11(const float* output, const uint& outputSize, const uint& netW, const uint& netH,
    const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo;
  
  // YOLOv11 output shape is typically [1, 84, 8400] -> DeepStream sees [84, 8400]
  // 84 channels: cx, cy, w, h, 80 class scores
  // 8400 anchors
  
  const uint numAnchors = 8400;
  const uint numChannels = 84;
  const uint numClasses = numChannels - 4;
  
  // Check if outputSize matches expectation
  if (outputSize != numAnchors * numChannels) {
      // Fallback or error? For now, assume standard YOLOv11n shape
      // If mismatch, it might be the old format, but we are fixing for v11
  }

  for (uint i = 0; i < numAnchors; ++i) {
    // Find class with max score
    float maxProb = 0.0f;
    int maxIndex = -1;
    
    for (uint c = 0; c < numClasses; ++c) {
        // output is [channels, anchors] -> [c, i]
        // index = c * numAnchors + i
        float prob = output[(4 + c) * numAnchors + i];
        if (prob > maxProb) {
            maxProb = prob;
            maxIndex = c;
        }
    }

    if (maxIndex == -1 || maxProb < preclusterThreshold[maxIndex]) {
      continue;
    }

    float cx = output[0 * numAnchors + i];
    float cy = output[1 * numAnchors + i];
    float w  = output[2 * numAnchors + i];
    float h  = output[3 * numAnchors + i];

    float x1 = cx - w / 2.0f;
    float y1 = cy - h / 2.0f;
    float x2 = x1 + w;
    float y2 = y1 + h;

    addBBoxProposal(x1, y1, x2, y2, netW, netH, maxIndex, maxProb, binfo);
  }

  return binfo;
}

static bool
NvDsInferParseCustomYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  std::vector<NvDsInferParseObjectInfo> objects;

  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  
  // Check dimensions to decide parsing method
  // YOLOv11: [84, 8400] -> size = 705600
  // Old decoded format: [N, 6] -> size = N*6
  
  const uint outputSize = output.inferDims.numElements;
  
  // Heuristic: if size is large (like 84*8400), use v11 parser
  if (outputSize == 84 * 8400) {
      std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorYolo11(
          (const float*) (output.buffer), outputSize,
          networkInfo.width, networkInfo.height, detectionParams.perClassPreclusterThreshold);
      objects.insert(objects.end(), outObjs.begin(), outObjs.end());
  } else {
      // Fallback to old parser (assumes decoded boxes)
      // Note: output.inferDims.d[0] might be misleading if dims are collapsed
      // We use outputSize / 6 as number of boxes
      const uint numBoxes = outputSize / 6;
      std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorYolo(
          (const float*) (output.buffer), numBoxes,
          networkInfo.width, networkInfo.height, detectionParams.perClassPreclusterThreshold);
      objects.insert(objects.end(), outObjs.begin(), outObjs.end());
  }

  objectList = objects;

  return true;
}

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
