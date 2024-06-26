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

#pragma once

#include "parsed_expr.h"

namespace infinity {

class BetweenExpr : public ParsedExpr {
public:
    explicit BetweenExpr() : ParsedExpr(ParsedExprType::kBetween) {}

    ~BetweenExpr() override;

    [[nodiscard]] std::string ToString() const override;

public:
    ParsedExpr *value_{nullptr};
    ParsedExpr *upper_bound_{nullptr};
    ParsedExpr *lower_bound_{nullptr};
};

} // namespace infinity
