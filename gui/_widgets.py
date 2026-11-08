"""Small Streamlit widget helpers."""

from typing import Any, Tuple, Union

import streamlit as st


def int_slider(label: str, min_value: int, max_value: int, value: Union[int, Tuple[int, int]], **kwargs: Any):
    """
    st.slider that also works when there is only one possible value: Streamlit raises
    StreamlitInvalidMinMaxError for min_value == max_value (e.g. a document with a single chunk), so the only
    value (or the range (min, max)) is returned with a caption instead.
    """
    if max_value <= min_value:
        st.caption(f"{label}: {min_value} (only one available)")
        return (min_value, min_value) if isinstance(value, tuple) else min_value
    return st.slider(label, min_value=min_value, max_value=max_value, value=value, **kwargs)
