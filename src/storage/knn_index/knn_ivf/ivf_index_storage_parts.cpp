// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module;

#include <cassert>
#include <vector>
module ivf_index_storage;

import stl;
import infinity_exception;
import status;
import logger;
import third_party;
import index_ivf;
import column_vector;
import internal_types;
import logical_type;
import data_type;
import kmeans_partition;
import search_top_1;
import search_top_k;
import column_vector;
import knn_scan_data;
import ivf_index_util_func;

namespace infinity {

class IVF_Part_Storage {
    u32 part_id_ = std::numeric_limits<u32>::max();
    u32 embedding_dimension_ = 0;

protected:
    u32 embedding_num_ = 0;
    Vector<SegmentOffset> embedding_segment_offsets_ = {};

public:
    IVF_Part_Storage(const u32 part_id, const u32 embedding_dimension) : part_id_(part_id), embedding_dimension_(embedding_dimension) {}
    virtual ~IVF_Part_Storage() = default;
    static UniquePtr<IVF_Part_Storage>
    Make(u32 part_id, u32 embedding_dimension, EmbeddingDataType embedding_data_type, const IndexIVFStorageOption &ivf_storage_option);
    u32 part_id() const { return part_id_; }
    u32 embedding_dimension() const { return embedding_dimension_; }
    u32 embedding_num() const { return embedding_num_; }
    SegmentOffset embedding_segment_offset(const u32 embedding_index) const { return embedding_segment_offsets_[embedding_index]; }

    virtual void Save(LocalFileHandle &file_handle) const {
        file_handle.Append(&part_id_, sizeof(part_id_));
        file_handle.Append(&embedding_dimension_, sizeof(embedding_dimension_));
        file_handle.Append(&embedding_num_, sizeof(embedding_num_));
        static_assert(std::is_same_v<SegmentOffset, typename decltype(embedding_segment_offsets_)::value_type>);
        assert(embedding_num_ == embedding_segment_offsets_.size());
        file_handle.Append(embedding_segment_offsets_.data(), embedding_num_ * sizeof(SegmentOffset));
    }

    virtual void Load(LocalFileHandle &file_handle) {
        file_handle.Read(&part_id_, sizeof(part_id_));
        file_handle.Read(&embedding_dimension_, sizeof(embedding_dimension_));
        file_handle.Read(&embedding_num_, sizeof(embedding_num_));
        embedding_segment_offsets_.resize(embedding_num_);
        file_handle.Read(embedding_segment_offsets_.data(), embedding_num_ * sizeof(SegmentOffset));
    }

    virtual void AppendOneEmbedding(const void *embedding_ptr, SegmentOffset segment_offset, const IVF_Centroids_Storage *ivf_centroids_storage) = 0;

    virtual void SearchIndex(const KnnDistanceBase1 *knn_distance,
                             const void *query_ptr,
                             EmbeddingDataType query_element_type,
                             const std::function<bool(SegmentOffset)> &satisfy_filter_func,
                             const std::function<void(f32, SegmentOffset)> &add_result_func) const = 0;

    // only for unit-test, return f32 / i8 / u8 embedding data
    virtual Pair<const void *, SharedPtr<void>> GetDataForTest(u32 embedding_id) const = 0;
};

template <EmbeddingDataType plain_data_type, EmbeddingDataType src_embedding_data_type>
class IVF_Part_Storage_Plain final : public IVF_Part_Storage {
    using StorageDataT = EmbeddingDataTypeToCppTypeT<plain_data_type>;
    static_assert(IsAnyOf<StorageDataT, i8, u8, f32, Float16T, BFloat16T>);
    using ColumnEmbeddingElementT = EmbeddingDataTypeToCppTypeT<src_embedding_data_type>;
    static_assert(IsAnyOf<ColumnEmbeddingElementT, i8, u8, f64, f32, Float16T, BFloat16T>);
    static_assert(std::is_same_v<StorageDataT, ColumnEmbeddingElementT> ||
                  (!IsAnyOf<StorageDataT, i8, u8> && !IsAnyOf<ColumnEmbeddingElementT, i8, u8>));

    Vector<StorageDataT> data_{};

public:
    IVF_Part_Storage_Plain(const u32 part_id, const u32 embedding_dimension) : IVF_Part_Storage(part_id, embedding_dimension) {}

    void Save(LocalFileHandle &file_handle) const override {
        IVF_Part_Storage::Save(file_handle);
        const u32 element_cnt = embedding_num() * embedding_dimension();
        assert(element_cnt == data_.size());
        file_handle.Append(data_.data(), element_cnt * sizeof(StorageDataT));
    }
    void Load(LocalFileHandle &file_handle) override {
        IVF_Part_Storage::Load(file_handle);
        const u32 element_cnt = embedding_num() * embedding_dimension();
        data_.resize(element_cnt);
        file_handle.Read(data_.data(), element_cnt * sizeof(StorageDataT));
    }

    void AppendOneEmbedding(const void *embedding_ptr, const SegmentOffset segment_offset, const IVF_Centroids_Storage *) override {
        const auto *src_embedding_data = static_cast<const ColumnEmbeddingElementT *>(embedding_ptr);
        if constexpr (std::is_same_v<StorageDataT, ColumnEmbeddingElementT>) {
            data_.insert(data_.end(), src_embedding_data, src_embedding_data + embedding_dimension());
        } else {
            static_assert(IsAnyOf<StorageDataT, f32, Float16T, BFloat16T> && IsAnyOf<ColumnEmbeddingElementT, f64, f32, Float16T, BFloat16T>);
            data_.reserve(data_.size() + embedding_dimension());
            for (u32 i = 0; i < embedding_dimension(); ++i) {
                if constexpr (std::is_same_v<f64, ColumnEmbeddingElementT>) {
                    data_.push_back(static_cast<StorageDataT>(static_cast<f32>(src_embedding_data[i])));
                } else {
                    data_.push_back(static_cast<StorageDataT>(src_embedding_data[i]));
                }
            }
        }
        embedding_segment_offsets_.push_back(segment_offset);
        ++embedding_num_;
    }

    void SearchIndex(const KnnDistanceBase1 *knn_distance,
                     const void *query_ptr,
                     const EmbeddingDataType query_element_type,
                     const std::function<bool(SegmentOffset)> &satisfy_filter_func,
                     const std::function<void(f32, SegmentOffset)> &add_result_func) const override {
        auto ReturnT = [&]<EmbeddingDataType query_element_type> {
            if constexpr ((query_element_type == EmbeddingDataType::kElemFloat && IsAnyOf<ColumnEmbeddingElementT, f64, f32, Float16T, BFloat16T>) ||
                          (query_element_type == src_embedding_data_type &&
                           (query_element_type == EmbeddingDataType::kElemInt8 || query_element_type == EmbeddingDataType::kElemUInt8))) {
                return SearchIndexT<query_element_type>(knn_distance,
                                                        static_cast<const EmbeddingDataTypeToCppTypeT<query_element_type> *>(query_ptr),
                                                        satisfy_filter_func,
                                                        add_result_func);

            } else {
                UnrecoverableError("Invalid Query EmbeddingDataType");
            }
        };
        switch (query_element_type) {
            case EmbeddingDataType::kElemFloat: {
                return ReturnT.template operator()<EmbeddingDataType::kElemFloat>();
            }
            case EmbeddingDataType::kElemUInt8: {
                return ReturnT.template operator()<EmbeddingDataType::kElemUInt8>();
            }
            case EmbeddingDataType::kElemInt8: {
                return ReturnT.template operator()<EmbeddingDataType::kElemInt8>();
            }
            default: {
                UnrecoverableError("Invalid EmbeddingDataType");
            }
        }
    }

    template <EmbeddingDataType query_element_type>
    void SearchIndexT(const KnnDistanceBase1 *knn_distance,
                      const EmbeddingDataTypeToCppTypeT<query_element_type> *query_ptr,
                      const std::function<bool(SegmentOffset)> &satisfy_filter_func,
                      const std::function<void(f32, SegmentOffset)> &add_result_func) const {
        using QueryDataType = EmbeddingDataTypeToCppTypeT<query_element_type>;
        auto knn_distance_1 = dynamic_cast<const KnnDistance1<QueryDataType, f32> *>(knn_distance);
        if (!knn_distance_1) [[unlikely]] {
            UnrecoverableError("Invalid KnnDistance1");
        }
        auto dist_func = knn_distance_1->dist_func_;
        const auto total_embedding_num = embedding_num();
        for (u32 i = 0; i < total_embedding_num; ++i) {
            const auto segment_offset = embedding_segment_offset(i);
            if (!satisfy_filter_func(segment_offset)) {
                continue;
            }
            auto v_ptr = data_.data() + i * embedding_dimension();
            auto [calc_ptr, _] = GetSearchCalcPtr<QueryDataType>(v_ptr, embedding_dimension());
            auto d = dist_func(calc_ptr, query_ptr, embedding_dimension());
            add_result_func(d, segment_offset);
        }
    }

    // only for unit-test, return f32 / i8 / u8 embedding data
    Pair<const void *, SharedPtr<void>> GetDataForTest(const u32 embedding_id) const override {
        if constexpr (IsAnyOf<StorageDataT, i8, u8, f32>) {
            return {data_.data() + embedding_id * embedding_dimension(), nullptr};
        } else if constexpr (IsAnyOf<StorageDataT, Float16T, BFloat16T>) {
            auto tmp_data = MakeUniqueForOverwrite<f32[]>(embedding_dimension());
            const auto *start_ptr = data_.data() + embedding_id * embedding_dimension();
            for (u32 i = 0; i < embedding_dimension(); ++i) {
                tmp_data[i] = static_cast<f32>(start_ptr[i]);
            }
            Pair<const void *, SharedPtr<void>> result(tmp_data.get(),
                                                       SharedPtr<void>(tmp_data.get(), [](void *ptr) { delete[] static_cast<f32 *>(ptr); }));
            tmp_data.release();
            return result;
        } else {
            static_assert(false, "unexpected type");
            return {};
        }
    }
};

UniquePtr<IVF_Part_Storage>
IVF_Part_Storage::Make(u32 part_id, u32 embedding_dimension, EmbeddingDataType embedding_data_type, const IndexIVFStorageOption &ivf_storage_option) {
    switch (ivf_storage_option.type_) {
        case IndexIVFStorageOption::Type::kPlain: {
            auto GetPlainResult =
                [part_id, embedding_dimension, embedding_data_type]<EmbeddingDataType plain_data_type>() -> UniquePtr<IVF_Part_Storage> {
                return ApplyEmbeddingDataTypeToFunc(
                    embedding_data_type,
                    [part_id, embedding_dimension]<EmbeddingDataType src_embedding_data_type>() -> UniquePtr<IVF_Part_Storage> {
                        auto not_i8_u8 = [](EmbeddingDataType t) consteval {
                            return t != EmbeddingDataType::kElemInt8 && t != EmbeddingDataType::kElemUInt8;
                        };
                        if constexpr (plain_data_type == src_embedding_data_type ||
                                      (not_i8_u8(plain_data_type) && not_i8_u8(src_embedding_data_type))) {
                            return MakeUnique<IVF_Part_Storage_Plain<plain_data_type, src_embedding_data_type>>(part_id, embedding_dimension);
                        } else {
                            return nullptr;
                        }
                    },
                    [] { return UniquePtr<IVF_Part_Storage>(); });
            };
            switch (ivf_storage_option.plain_storage_data_type_) {
                case EmbeddingDataType::kElemInt8: {
                    return GetPlainResult.template operator()<EmbeddingDataType::kElemInt8>();
                }
                case EmbeddingDataType::kElemUInt8: {
                    return GetPlainResult.template operator()<EmbeddingDataType::kElemUInt8>();
                }
                case EmbeddingDataType::kElemFloat: {
                    return GetPlainResult.template operator()<EmbeddingDataType::kElemFloat>();
                }
                case EmbeddingDataType::kElemFloat16: {
                    return GetPlainResult.template operator()<EmbeddingDataType::kElemFloat16>();
                }
                case EmbeddingDataType::kElemBFloat16: {
                    return GetPlainResult.template operator()<EmbeddingDataType::kElemBFloat16>();
                }
                case EmbeddingDataType::kElemDouble:
                case EmbeddingDataType::kElemBit:
                case EmbeddingDataType::kElemInt16:
                case EmbeddingDataType::kElemInt32:
                case EmbeddingDataType::kElemInt64:
                case EmbeddingDataType::kElemInvalid: {
                    UnrecoverableError("Invalid IVF plain_data_type");
                    return {};
                }
            }
        }
        case IndexIVFStorageOption::Type::kScalarQuantization: {
            UnrecoverableError("Not implemented");
            return {};
        }
        case IndexIVFStorageOption::Type::kProductQuantization: {
            UnrecoverableError("Not implemented");
            return {};
        }
    }
}

UniquePtr<IVF_Parts_Storage> IVF_Parts_Storage::Make(const u32 embedding_dimension,
                                                     const u32 centroids_num,
                                                     const EmbeddingDataType embedding_data_type,
                                                     const IndexIVFStorageOption &ivf_storage_option) {
    // TODO
}

} // namespace infinity
