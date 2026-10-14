import os, sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import enzymeadvisor_engine as ea
import enzymeadvisor_chat as ch

ea.MAX_LEN = 600       # keeps memory use low on free hosting

EXAMPLE_SEQ = 'MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW'
EXAMPLE_MEAS = 'L199E 1.251, R59S 0.149, I171Y 1.118, N152R 1.260, L146T 0.079, G248T 0.002, T139Y 0.426, L150E 0.007, L73K 0.002, I206V 1.132, L197S 0.142, D83I 0.025, T138F 0.852, T139H 0.723, T107N 0.039, E175L 1.109, E169S 1.137, L205N 0.002, T187D 1.013, D33S 0.852'

st.set_page_config(page_title="EnzymeAdvisor", page_icon="🧬")
st.title("EnzymeAdvisor")
st.caption(
    "Choose which enzyme mutations to test when your lab budget is small. Every reply is built from model "
    "numbers; no language model writes the explanations. These are model estimates, not measurements. The method "
    "was tested retrospectively on 22 enzyme datasets, not in a wet lab. This is a public demo: do not enter "
    "confidential sequences. Type help for commands."
)

if "bot" not in st.session_state:          # one separate conversation per visitor
    st.session_state.bot = ch.EnzymeAdvisorChat()
    st.session_state.messages = []

clicked = None
with st.sidebar:
    st.header("Try the demo")
    st.write("Click in this order, and wait for each answer before the next click:")
    examples = [("1. budget 5", "budget 5"),
                ("2. set the demo protein", "sequence " + EXAMPLE_SEQ),
                ("3. recommend (ESM-2 only)", "recommend"),
                ("4. enter 20 demo measurements", "measured " + EXAMPLE_MEAS),
                ("5. recommend again (learning method)", "recommend"),
                ("6. why not N152F", "why not N152F"),
                ("7. self-check", "check"),
                ("help", "help")]
    for label, cmd in examples:
        if st.button(label, use_container_width=True):
            clicked = cmd
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.bot = ch.EnzymeAdvisorChat()
        st.session_state.messages = []
        st.rerun()

prompt = st.chat_input("Type a command, or paste a protein sequence")
text = clicked or prompt

history = st.container()
status = st.empty()
if text:
    status.info("Working... please wait for the answer before clicking again.")
    try:
        reply = st.session_state.bot.reply(text)
    except Exception as e:
        reply = f"Something went wrong ({type(e).__name__}: {e}). Use Reset conversation to start over."
    shown = text if len(text) < 300 else text[:300] + " ..."
    # saved right away, with no page calls in between, so a new click cannot lose this exchange
    st.session_state.messages.append({"role": "user", "content": shown})
    st.session_state.messages.append({"role": "assistant", "content": reply})
    status.empty()

with history:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.text(m["content"])
