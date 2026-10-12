# EnzymeAdvisor

A decision-support tool for choosing which enzyme mutations to test when the lab budget is small.

- With no measurements it ranks single mutations by ESM-2 (esm2_t12_35M_UR50D).
- With 20 or more measured variants it fits a Gaussian process, adds an uncertainty bonus (beta = 2)
  and applies a near-duplicate rule so a batch is spread over different positions.
- Every explanation is built from the same numbers used to choose the panel; no language model writes text.
  A self-check recomputes those numbers independently.

Retrospective benchmark (22 enzyme datasets, small budgets): about twice as many high-fitness variants as random
picking, and about the same best variant as ESM-2 ranking. Not validated in a wet lab; the model mainly identifies
promising positions, not specific substitutions. Measured values must follow higher = better.

Run it locally:

    pip install -r requirements.txt
    streamlit run streamlit_app.py

Files: streamlit_app.py (web page), enzymeadvisor_chat.py (chat commands), enzymeadvisor_engine.py (engine).
