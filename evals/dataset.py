"""Golden questions for the retrieval eval.

Each case names the document, the question, and the strings that must appear
in a retrieved chunk for the retrieval to count as correct. Needles are the
literal text as it survives extraction - not as it reads in the PDF - so a
case fails loudly when an extraction regression mangles it.

`answer_needles` are checked against the generated answer instead, and may
differ from the retrieval needles: an answer says "0.9", the chunk holding
it may say "β1 = 0.9".
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    doc: str
    question: str
    needles: tuple[str, ...]
    answer_needles: tuple[str, ...] = field(default=())
    # Known-failing cases stay in the set rather than being deleted: a
    # limitation you have measured is worth more than one you have hidden.
    expected_fail: str = ""

    @property
    def answer_targets(self) -> tuple[str, ...]:
        return self.answer_needles or self.needles


CASES: tuple[Case, ...] = (
    # --- attention.pdf: prose ------------------------------------------
    Case(
        "attention.pdf",
        "How many layers are in the encoder and decoder stacks?",
        ("N = 6",),
        ("6",),
    ),
    Case(
        "attention.pdf",
        "What are the two sub-layers inside each encoder layer?",
        ("Feed Forward", "feed-forward"),
        # The paper says "multi-head self-attention" and "position-wise fully
        # connected feed-forward network"; a needle of "Multi-Head Attention"
        # failed a demonstrably correct answer. Match the stems.
        ("multi-head", "feed-forward"),
    ),
    # --- attention.pdf: numbers that only survive PUA normalisation ----
    Case(
        "attention.pdf",
        "What Adam beta1 and beta2 values were used for training?",
        ("0.98",),
        ("0.9", "0.98"),
    ),
    # --- attention.pdf: table bodies ------------------------------------
    Case(
        "attention.pdf",
        "What F1 score did the Transformer reach on WSJ section 23 parsing?",
        ("91.3",),
        ("91.3",),
    ),
    Case(
        "attention.pdf",
        "What BLEU score did the big Transformer achieve on English-to-German?",
        ("28.4",),
        ("28.4",),
    ),
    Case(
        "attention.pdf",
        "How many parameters and what dropout did the base model use?",
        ("0.1",),
    ),
    # --- attention.pdf: figure-derived ----------------------------------
    Case(
        "attention.pdf",
        "What does the model architecture diagram show on its left and right halves?",
        ("raster image, page 3", "encoder"),
        ("encoder", "decoder"),
    ),
    # --- attention.pdf: the known vocabulary-mismatch failure -----------
    Case(
        "attention.pdf",
        "What problem does the Transformer architecture aim to solve?",
        ("precludes parallelization",),
        expected_fail=(
            "Vocabulary mismatch. The chunk states the problem in terms of "
            "'Recurrent models' and 'sequential nature' and never uses the "
            "words 'Transformer' or 'problem', so neither the dense nor the "
            "lexical arm matches it. Needs query expansion or a reranker."
        ),
    ),
    # --- absence: the honest answer is 'not in the document' ------------
    Case(
        "attention.pdf",
        "What optimizer learning rate did they use for training BERT?",
        ("Adam",),
        ("does not", "no information", "not contain"),
    ),
    # --- other documents ------------------------------------------------
    Case(
        "clean_text_1.pdf",
        "What value is associated with Gamma?",
        ("Gamma | 30", "Gamma"),
        ("30",),
    ),
    Case(
        "multi_page_mixed.pdf",
        "What value does the table give for Beta?",
        ("Beta",),
        ("20",),
    ),
    Case(
        "vector_chart_1.pdf",
        "Which category has the tallest bar in the chart?",
        ("vector graphic, page 1",),
        ("category b",),
    ),
    Case(
        "charts.pdf",
        "What is the forecast about and for which city?",
        ("Sevilla",),
        ("Sevilla",),
    ),
)
