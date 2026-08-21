from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import numpy as np
import pandas as pd


PERSIAN_LEAP_RESIDUES = {1, 5, 9, 13, 17, 22, 26, 30}
DATE_PATTERN = re.compile(
    r"^\s*(\d{4})/(\d{1,2})/(\d{1,2})(?:\s+(\d{1,2}):(\d{1,2}))?\s*$"
)

SIGNATURE_COLUMNS = [
    "sig_char_count",
    "sig_punct_count",
    "sig_question_count",
    "sig_excl_count",
    "sig_emoji_count",
    "sig_word_count",
    "sig_neg_count",
    "sig_pos_count",
    "sig_pos_emoji",
    "sig_neg_emoji",
]

POST_SIGNAL_COLUMNS = [
    "post_char_count",
    "post_punct_count",
    "post_question_count",
    "post_excl_count",
    "post_emoji_count",
    "post_pos_emoji",
    "post_neg_emoji",
    "post_word_count",
    "post_neg_count",
    "post_pos_count",
    "post_neg_count_temp",
    "post_pos_count_temp",
]

RAW_REQUIRED_COLUMNS = [
    "likes",
    "reply_to",
    "is_starter",
    "user_post_count",
    "gender",
    "posted_at",
    "join_date",
    "children_count",
    "category",
    "sub_category",
    "age_num",
    "education_clean",
    *SIGNATURE_COLUMNS,
    *POST_SIGNAL_COLUMNS,
]

CATEGORICAL_FEATURES = [
    "gender",
    "category",
    "sub_category",
    "education_clean",
    "posted_weekday_name",
    "posted_daypart",
]

FEATURE_ORDER = [
    "likes",
    "log1p_likes",
    "user_post_count",
    "log1p_user_post_count",
    "children_count",
    "age_missing",
    "age_num",
    "is_starter",
    "has_reply",
    "gender",
    "category",
    "sub_category",
    "education_clean",
    "posted_year",
    "posted_month",
    "posted_day",
    "posted_hour",
    "posted_minute",
    "posted_weekday",
    "posted_weekday_name",
    "posted_daypart",
    "posted_is_weekend",
    "posted_is_night",
    "posted_is_late_evening",
    "posted_hour_sin",
    "posted_hour_cos",
    "posted_month_sin",
    "posted_month_cos",
    "join_year",
    "join_month",
    "join_day",
    "account_age_days",
    "account_age_years",
    "posts_per_account_year",
    "log1p_posts_per_account_year",
    "likes_per_100_posts",
    "has_signature",
    "sig_char_count",
    "sig_punct_count",
    "sig_question_count",
    "sig_excl_count",
    "sig_emoji_count",
    "sig_word_count",
    "sig_neg_count",
    "sig_pos_count",
    "sig_pos_emoji",
    "sig_neg_emoji",
    "post_char_count",
    "post_punct_count",
    "post_question_count",
    "post_excl_count",
    "post_emoji_count",
    "post_pos_emoji",
    "post_neg_emoji",
    "post_word_count",
    "post_neg_count",
    "post_pos_count",
    "post_neg_count_temp",
    "post_pos_count_temp",
    "post_neg_word_ratio",
    "post_pos_word_ratio",
    "post_question_per_100_words",
    "post_excl_per_100_words",
    "post_emoji_per_100_words",
    "post_punct_per_100_chars",
    "post_temp_neg_ratio",
    "post_temp_pos_ratio",
]

FORBIDDEN_FEATURES = {
    "unique_post_id",
    "final_stress",
    "clinical_class",
    "stress_proxy",
    "model_role",
    "official_eval",
    "oof_fold",
    "group_id",
    "label_source",
    "annotation_weight",
    "class_weight",
    "training_sample_weight",
    "content",
    "thread_title",
    "author",
    "thread_id",
    "reply_to",
}


@dataclass
class FeatureEngineeringConfig:
    missing_category_token: str = "__MISSING__"
    minimum_valid_age: int = 12
    maximum_valid_age: int = 80
    replace_missing_signature_with_zero: bool = True
    use_raw_text: bool = False


class Member1FeatureEngineer:
    """Stateless, leakage-safe tabular feature engineering.

    Raw post text is never modeled. The predefined post/signature count columns are
    accepted as upstream numeric signals. The original lexicons used to produce
    positive/negative count columns were not included in the handoff, so future
    inference should provide those predefined columns from the same upstream
    extractor whenever exact reproduction is required.
    """

    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        self.config = config or FeatureEngineeringConfig()
        self.feature_order_ = list(FEATURE_ORDER)
        self.categorical_features_ = list(CATEGORICAL_FEATURES)
        self.numeric_features_ = [
            column for column in self.feature_order_
            if column not in self.categorical_features_
        ]

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_input_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input_columns(X)
        df = X.copy()
        out = pd.DataFrame(index=df.index)

        likes = pd.to_numeric(df["likes"], errors="coerce").clip(lower=0)
        user_posts = pd.to_numeric(df["user_post_count"], errors="coerce").clip(lower=0)
        children = pd.to_numeric(df["children_count"], errors="coerce")

        age_raw = pd.to_numeric(df["age_num"], errors="coerce")
        age_missing = age_raw.isna() | age_raw.eq(-1)
        valid_age = age_raw.mask(
            age_missing
            | age_raw.lt(self.config.minimum_valid_age)
            | age_raw.gt(self.config.maximum_valid_age),
            np.nan,
        )

        reply_numeric = pd.to_numeric(df["reply_to"], errors="coerce").fillna(-1)

        out["likes"] = likes
        out["log1p_likes"] = np.log1p(likes)
        out["user_post_count"] = user_posts
        out["log1p_user_post_count"] = np.log1p(user_posts)
        out["children_count"] = children
        out["age_missing"] = age_missing.astype(int)
        out["age_num"] = valid_age
        out["is_starter"] = df["is_starter"].map(self._bool_to_int).astype(int)
        out["has_reply"] = reply_numeric.ne(-1).astype(int)

        for column in ["gender", "category", "sub_category", "education_clean"]:
            out[column] = (
                df[column]
                .fillna(self.config.missing_category_token)
                .astype(str)
            )

        posted = [self._parse_jalali_datetime(value) for value in df["posted_at"]]
        joined = [self._parse_jalali_datetime(value) for value in df["join_date"]]

        posted_year = self._extract_date_part(posted, "year")
        posted_month = self._extract_date_part(posted, "month")
        posted_day = self._extract_date_part(posted, "day")
        posted_hour = self._extract_date_part(posted, "hour")
        posted_minute = self._extract_date_part(posted, "minute")
        posted_index = self._extract_date_part(posted, "day_index")

        join_year = self._extract_date_part(joined, "year")
        join_month = self._extract_date_part(joined, "month")
        join_day = self._extract_date_part(joined, "day")
        join_index = self._extract_date_part(joined, "day_index")

        weekday = (6 + posted_index) % 7
        weekday_map = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }

        out["posted_year"] = posted_year
        out["posted_month"] = posted_month
        out["posted_day"] = posted_day
        out["posted_hour"] = posted_hour
        out["posted_minute"] = posted_minute
        out["posted_weekday"] = weekday
        out["posted_weekday_name"] = (
            pd.Series(weekday, index=df.index)
            .map(weekday_map)
            .fillna(self.config.missing_category_token)
            .astype(str)
        )
        out["posted_daypart"] = [
            self._daypart(hour) for hour in posted_hour
        ]
        out["posted_is_weekend"] = np.where(
            np.isnan(weekday),
            np.nan,
            np.isin(weekday, [3, 4]).astype(float),
        )
        out["posted_is_night"] = np.where(
            np.isnan(posted_hour),
            np.nan,
            (posted_hour < 6).astype(float),
        )
        out["posted_is_late_evening"] = np.where(
            np.isnan(posted_hour),
            np.nan,
            (posted_hour >= 22).astype(float),
        )
        out["posted_hour_sin"] = np.sin(2 * np.pi * posted_hour / 24)
        out["posted_hour_cos"] = np.cos(2 * np.pi * posted_hour / 24)
        out["posted_month_sin"] = np.sin(2 * np.pi * (posted_month - 1) / 12)
        out["posted_month_cos"] = np.cos(2 * np.pi * (posted_month - 1) / 12)

        out["join_year"] = join_year
        out["join_month"] = join_month
        out["join_day"] = join_day

        account_age_days = posted_index - join_index
        account_age_days = np.where(
            np.isnan(account_age_days),
            np.nan,
            np.maximum(account_age_days, 0),
        )
        account_age_years = account_age_days / 365.25
        post_rate = user_posts.to_numpy(dtype=float) / np.maximum(account_age_years, 0.25)
        post_rate = np.where(np.isnan(account_age_years), np.nan, post_rate)

        out["account_age_days"] = account_age_days
        out["account_age_years"] = account_age_years
        out["posts_per_account_year"] = post_rate
        out["log1p_posts_per_account_year"] = np.log1p(post_rate)
        out["likes_per_100_posts"] = likes / (1 + user_posts / 100)

        signature = df[SIGNATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        no_signature = signature.eq(-1).all(axis=1)
        out["has_signature"] = (~no_signature).astype(int)
        for column in SIGNATURE_COLUMNS:
            values = signature[column]
            if self.config.replace_missing_signature_with_zero:
                values = values.mask(values.eq(-1), 0)
            out[column] = values

        for column in POST_SIGNAL_COLUMNS:
            out[column] = pd.to_numeric(df[column], errors="coerce")

        word_count = out["post_word_count"].clip(lower=1)
        char_count = out["post_char_count"].clip(lower=1)
        out["post_neg_word_ratio"] = out["post_neg_count"] / word_count
        out["post_pos_word_ratio"] = out["post_pos_count"] / word_count
        out["post_question_per_100_words"] = 100 * out["post_question_count"] / word_count
        out["post_excl_per_100_words"] = 100 * out["post_excl_count"] / word_count
        out["post_emoji_per_100_words"] = 100 * out["post_emoji_count"] / word_count
        out["post_punct_per_100_chars"] = 100 * out["post_punct_count"] / char_count
        out["post_temp_neg_ratio"] = out["post_neg_count_temp"] / word_count
        out["post_temp_pos_ratio"] = out["post_pos_count_temp"] / word_count

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out[self.feature_order_]

        unexpected_forbidden = sorted(set(out.columns) & FORBIDDEN_FEATURES)
        if unexpected_forbidden:
            raise RuntimeError(
                f"Forbidden columns entered model matrix: {unexpected_forbidden}"
            )

        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(self.feature_order_, dtype=object)

    def to_config_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "feature_order": list(self.feature_order_),
            "categorical_features": list(self.categorical_features_),
            "numeric_features": list(self.numeric_features_),
            "raw_required_columns": list(RAW_REQUIRED_COLUMNS),
            "forbidden_features": sorted(FORBIDDEN_FEATURES),
        }

    def _validate_input_columns(self, X: pd.DataFrame) -> None:
        missing = [column for column in RAW_REQUIRED_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(
                "Input is missing required upstream metadata/count columns: "
                + ", ".join(missing)
            )

        forbidden_present = sorted(
            set(self.feature_order_) & FORBIDDEN_FEATURES
        )
        if forbidden_present:
            raise RuntimeError(
                f"Feature contract violation: {forbidden_present}"
            )

    @staticmethod
    def _bool_to_int(value) -> int:
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        return int(str(value).strip().lower() in {"1", "true", "yes", "y"})

    def _daypart(self, hour: float) -> str:
        if np.isnan(hour):
            return self.config.missing_category_token
        if hour < 6:
            return "overnight"
        if hour < 12:
            return "morning"
        if hour < 18:
            return "afternoon"
        return "evening"

    @staticmethod
    def _extract_date_part(parsed: Iterable[Optional[dict]], key: str) -> np.ndarray:
        return np.asarray(
            [item[key] if item is not None else np.nan for item in parsed],
            dtype=float,
        )

    @staticmethod
    def _is_persian_leap(year: int) -> bool:
        return int(year) % 33 in PERSIAN_LEAP_RESIDUES

    @classmethod
    def _persian_days_before_year(cls, year: int, anchor: int = 1400) -> int:
        year = int(year)
        days = 0
        if year >= anchor:
            for current_year in range(anchor, year):
                days += 366 if cls._is_persian_leap(current_year) else 365
        else:
            for current_year in range(year, anchor):
                days -= 366 if cls._is_persian_leap(current_year) else 365
        return days

    @classmethod
    def _persian_day_index(cls, year: int, month: int, day: int) -> float:
        try:
            year, month, day = int(year), int(month), int(day)
        except (TypeError, ValueError):
            return np.nan

        if not (1200 <= year <= 1600 and 1 <= month <= 12):
            return np.nan

        month_lengths = [31] * 6 + [30] * 5 + [
            30 if cls._is_persian_leap(year) else 29
        ]
        if not (1 <= day <= month_lengths[month - 1]):
            return np.nan

        return float(
            cls._persian_days_before_year(year)
            + sum(month_lengths[: month - 1])
            + day
            - 1
        )

    @classmethod
    def _parse_jalali_datetime(cls, value) -> Optional[dict]:
        if pd.isna(value):
            return None
        match = DATE_PATTERN.match(str(value))
        if not match:
            return None

        year, month, day, hour, minute = match.groups()
        year, month, day = int(year), int(month), int(day)
        hour, minute = int(hour or 0), int(minute or 0)
        day_index = cls._persian_day_index(year, month, day)

        if np.isnan(day_index) or not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None

        return {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "day_index": day_index,
        }
