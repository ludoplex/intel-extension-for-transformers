//  Copyright (c) 2021 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "llga_kernel.hpp"

#include "common.hpp"

namespace executor {

void LLGAKernel::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // set dtype of output according to partition's logical tensors
  vector<logical_tensor> outputs = partition_.get_out_ports();
  for (size_t idx = 0; idx < outputs.size(); ++idx) {
    string output_dtype = "fp32";
    if (outputs[idx].get_data_type() == data_type::u8)
      output_dtype = "u8";
    else if (outputs[idx].get_data_type() == data_type::s8)
      output_dtype = "s8";
    output[idx]->set_dtype(output_dtype);
  }
  return;
}

int GetTensorIndexFromID(const vector<logical_tensor>& lts, int id) {
  for (int i = 0; i < lts.size(); i++) {
    if (lts[i].get_id() == id) {
      return i;
    }
  }
  return -1;
}

void LLGAKernel::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // compile partition and query output shape
  inputs_lt = partition_.get_in_ports();
  outputs_lt = partition_.get_out_ports();
  for (size_t idx = 0; idx < inputs_lt.size(); ++idx) {
    size_t id = inputs_lt[idx].get_id();
    logical_tensor temp_lt = model_->name2lts_[model_->id2names_[id]];
    try {
      temp_lt.get_dims();
      inputs_lt[idx] = temp_lt;
    } catch (...) {
      inputs_lt[idx] = logical_tensor {temp_lt.get_id(), temp_lt.get_data_type(),
                                    input[idx]->shape(), layout_type::strided};
      model_->name2lts_[model_->id2names_[id]] = inputs_lt[idx];
    }
    inputs_ts.push_back(dnnl::graph::tensor {inputs_lt[idx], model_->eng_, nullptr});
  }
  cp_ = partition_.compile(inputs_lt, outputs_lt, model_->eng_);
  for (size_t idx = 0; idx < outputs_lt.size(); ++idx) {
    size_t id = outputs_lt[idx].get_id();
    auto query_lt = cp_.query_logical_tensor(id);
    model_->name2lts_[model_->id2names_[id]] = logical_tensor {id, outputs_lt[idx].get_data_type(),
                                                               query_lt.get_dims(), layout_type::strided};
    output[idx]->set_shape(query_lt.get_dims());
    outputs_ts.push_back(dnnl::graph::tensor {query_lt, model_->eng_, nullptr});
  }

  // check whether there is an inplace operator.
  auto inplace_ports = cp_.get_inplace_ports();
  if (!inplace_ports.empty()) {
    inplace_ = true;
    auto src_id = inplace_ports[0].first;
    auto dst_id = inplace_ports[0].second;
    int src_idx = GetTensorIndexFromID(inputs_lt, src_id);
    assert(src_idx < inputs_lt.size() && src_idx >= 0);
    int dst_idx = GetTensorIndexFromID(outputs_lt, dst_id);

    inplace_index_ = std::make_pair(src_idx, dst_idx);
  }
}

void LLGAKernel::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (inplace_ && input[inplace_index_.first]->left_life() == 1) {
    auto data = input[inplace_index_.first]->mutable_data();
    input[inplace_index_.first]->unref_data(true);
    output[inplace_index_.second]->set_data(data);
  }

  for (int idx = 0; idx < inputs_lt.size(); idx++) {
    if (inplace_ && idx == inplace_index_.first) {
      inputs_ts[idx].set_data_handle(output[inplace_index_.second]->mutable_data());
    } else {
      inputs_ts[idx].set_data_handle(input[idx]->mutable_data());
    }
  }
  for (int idx = 0; idx < outputs_lt.size(); idx++) {
    outputs_ts[idx].set_data_handle(output[idx]->mutable_data());
  }
  cp_.execute(model_->strm_, inputs_ts, outputs_ts);
  model_->strm_.wait();

  if (inplace_) {
    for (int i = 0; i < input.size(); i++) {
      if (i = inplace_index_.first)
        continue;
      input[i]->unref_data();
    }
  } else {
    this->unref_tensors(input);
  }
}

}  // namespace executor
