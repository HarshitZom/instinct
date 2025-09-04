import math
from typing import List, Tuple, Optional
import warnings
from collections import Counter
from eval_suite.codebleu_utils import (
    _get_ngrams,
    _get_ngrams_with_weights,
    _extract_ast_nodes,
    _compute_lcs_length,
    _get_node_weight,
    LANGUAGE_EXTENSIONS,
    TOKEN_WEIGHTS,
)

try:
    from tree_sitter import Language, Parser, Node
except ImportError:
    raise ImportError("tree-sitter is required. Install with: pip install tree-sitter")


class TreeSitterManager:
    """Manages Tree-sitter parsers for different languages."""

    def __init__(self):
        self.parsers = {}
        self.languages = {}
        self._load_languages()

    def _load_languages(self):
        """Load available Tree-sitter languages."""
        try:
            self._load_python()
        except Exception as e:
            print(f"Warning: Could not load Python parser: {e}")

        try:
            self._load_javascript()
        except Exception as e:
            print(f"Warning: Could not load JavaScript parser: {e}")

        try:
            self._load_typescript()
        except Exception as e:
            print(f"Warning: Could not load TypeScript parser: {e}")

        try:
            self._load_java()
        except Exception as e:
            print(f"Warning: Could not load Java parser: {e}")

        try:
            self._load_c()
        except Exception as e:
            print(f"Warning: Could not load C parser: {e}")

        try:
            self._load_cpp()
        except Exception as e:
            print(f"Warning: Could not load C++ parser: {e}")

        try:
            self._load_rust()
        except Exception as e:
            print(f"Warning: Could not load Rust parser: {e}")

    def _load_python(self):
        """Load Python language using the tree-sitter-python package."""
        try:
            import tree_sitter_python

            language = Language(tree_sitter_python.language())
            self.languages["python"] = language

            parser = Parser(language)
            self.parsers["python"] = parser
        except ImportError:
            print(
                "tree-sitter-python package not installed. Try: pip install tree-sitter-python"
            )

    def _load_javascript(self):
        """Load JavaScript language using the tree-sitter-javascript package."""
        try:
            import tree_sitter_javascript

            language = Language(tree_sitter_javascript.language())
            self.languages["javascript"] = language

            parser = Parser(language)
            self.parsers["javascript"] = parser
        except ImportError:
            print(
                "tree-sitter-javascript package not installed. Try: pip install tree-sitter-javascript"
            )

    def _load_typescript(self):
        """Load TypeScript language using the tree-sitter-typescript package."""
        try:
            import tree_sitter_typescript

            if hasattr(tree_sitter_typescript, "language_typescript"):
                language = Language(tree_sitter_typescript.language_typescript())
                self.languages["typescript"] = language

                parser = Parser(language)
                self.parsers["typescript"] = parser
            elif hasattr(tree_sitter_typescript, "language_tsx"):
                # Use TSX as a fallback
                language = Language(tree_sitter_typescript.language_tsx())
                self.languages["typescript"] = language

                parser = Parser(language)
                self.parsers["typescript"] = parser

                print("Loaded TSX parser for TypeScript (fallback)")
            else:
                print("TypeScript parser could not be loaded. Package structure:")
                for attribute in dir(tree_sitter_typescript):
                    if not attribute.startswith("__"):
                        print(f"  {attribute}")

        except ImportError:
            print(
                "tree-sitter-typescript package not installed. Try: pip install tree-sitter-typescript"
            )
        except Exception as e:
            print(f"Error loading TypeScript parser: {e}")

    def _load_java(self):
        """Load Java language using the tree-sitter-java package."""
        try:
            import tree_sitter_java

            language = Language(tree_sitter_java.language())
            self.languages["java"] = language

            parser = Parser(language)
            self.parsers["java"] = parser
        except ImportError:
            print(
                "tree-sitter-java package not installed. Try: pip install tree-sitter-java"
            )

    def _load_c(self):
        """Load C language using the tree-sitter-c package."""
        try:
            import tree_sitter_c

            try:
                language = Language(tree_sitter_c.language())
                self.languages["c"] = language

                parser = Parser(language)
                self.parsers["c"] = parser
            except Exception as e:
                if "Incompatible Language version" in str(e):
                    print(f"Warning: Could not load C parser: {e}")
                    print(
                        "This is likely due to a version mismatch. You might need to rebuild the grammar."
                    )
                else:
                    raise
        except ImportError:
            print("tree-sitter-c package not installed. Try: pip install tree-sitter-c")

    def _load_cpp(self):
        """Load C++ language using the tree-sitter-cpp package."""
        try:
            import tree_sitter_cpp

            language = Language(tree_sitter_cpp.language())
            self.languages["cpp"] = language

            parser = Parser(language)
            self.parsers["cpp"] = parser
        except ImportError:
            print(
                "tree-sitter-cpp package not installed. Try: pip install tree-sitter-cpp"
            )

    def _load_rust(self):
        """Load Rust language using the tree-sitter-rust package."""
        try:
            import tree_sitter_rust

            language = Language(tree_sitter_rust.language())
            self.languages["rust"] = language

            parser = Parser(language)
            self.parsers["rust"] = parser
        except ImportError:
            print(
                "tree-sitter-rust package not installed. Try: pip install tree-sitter-rust"
            )

    def get_parser(self, lang: str) -> Optional[Parser]:
        """Get parser for the specified language."""
        normalized_lang = LANGUAGE_EXTENSIONS.get(lang.lower(), lang.lower())
        return self.parsers.get(normalized_lang)

    def is_supported(self, lang: str) -> bool:
        """Check if language is supported."""
        normalized_lang = LANGUAGE_EXTENSIONS.get(lang.lower(), lang.lower())
        return normalized_lang in self.parsers


# Global Tree-sitter manager instance
ts_manager = TreeSitterManager()


def calculate_codebleu(
    reference: str,
    hypothesis: str,
    lang: str,
) -> float:
    """
    Compute the CodeBLEU score between two code snippets.
    Args:
        reference: The reference (ground truth) code snippet
        hypothesis: The hypothesis (generated) code snippet
        lang: Language identifier (e.g., "py", "java", "js", "ts")
        weights: Weights for (n-gram, weighted n-gram, syntactic, semantic) components
    Returns:
        CodeBLEU score withOUT DFG ranging from 0 to 1

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If parsing fails
    """

    # Validate inputs
    if not reference or not reference.strip():
        warnings.warn("Reference code cannot be empty")
        return 0.0
    if not hypothesis or not hypothesis.strip():
        warnings.warn("Hypothesis code cannot be empty")
        return 0.0

    if not ts_manager.is_supported(lang):
        # raise ValueError(f"Language '{lang}' is not supported")
        warnings.warn(f"Language {lang}, not supported, returning 0.0")
        return 0.0

    try:
        # Compute component scores
        ngram_score = compute_ngram_bleu(reference, hypothesis)
        weighted_ngram_score = compute_weighted_ngram_bleu(reference, hypothesis, lang)
        ast_score = compute_ast_match(reference, hypothesis, lang)
        # dataflow_score = compute_dataflow_match(reference, hypothesis, lang)

        # Combine scores with weights
        no_dfg_codebleu_score = (
            0.333 * ngram_score + 0.333 * weighted_ngram_score + 0.334 * ast_score
        )

        # codebleu_score = 0.75 * no_dfg_codebleu_score + 0.25 * dataflow_score

        # Ensure score is in [0, 1]
        # return max(0.0, min(1.0, codebleu_score)), max(0.0, min(1.0, no_dfg_codebleu_score))
        return max(0.0, min(1.0, no_dfg_codebleu_score))

    except Exception as e:
        raise RuntimeError(f"Failed to compute CodeBLEU score: {e}")


def compute_ngram_bleu(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    """
    Compute the n-gram match score (standard BLEU) between code snippets.

    Args:
        reference: Reference code snippet
        hypothesis: Hypothesis code snippet
        max_n: Maximum n-gram order
        weights: Weights for n-grams from 1 to max_n

    Returns:
        BLEU score based on n-gram precision with brevity penalty
    """
    if not reference or not hypothesis:
        raise ValueError("Reference and hypothesis cannot be empty")
    if max_n < 1:
        raise ValueError("max_n must be at least 1")
    if len(weights) != max_n:
        raise ValueError(f"Weights length ({len(weights)}) must match max_n ({max_n})")
    if abs(sum(weights)) < 1e-9:
        raise ValueError("Weights cannot sum to zero")

    # Check for exact match first
    if reference == hypothesis:
        return 1.0

    try:
        ref_tokens = tokenize_code(reference, lang="python")  # Default to Python
        hyp_tokens = tokenize_code(hypothesis, lang="python")
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize code: {e}")

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # If tokens match exactly, return 1.0
    if ref_tokens == hyp_tokens:
        return 1.0

    # Compute n-gram precisions
    precisions = []
    has_matches = False

    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        hyp_ngrams = _get_ngrams(hyp_tokens, n)
        if not hyp_ngrams:
            precisions.append(1e-9)  # Small non-zero value to avoid log(0)
            continue

        # Count matches
        matches = 0
        ref_counts = Counter(ref_ngrams)
        hyp_counts = Counter(hyp_ngrams)

        for ngram in hyp_counts:
            match_count = min(hyp_counts[ngram], ref_counts[ngram])
            matches += match_count
            if match_count > 0:
                has_matches = True

        precision = matches / len(hyp_ngrams) if hyp_ngrams else 1e-9
        precisions.append(max(precision, 1e-9))  # Ensure non-zero

    if not has_matches:
        return 0.0

    # Compute brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)

    if hyp_len >= ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

    # Compute BLEU score
    log_precision_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
    bleu_score = brevity_penalty * math.exp(log_precision_sum)

    return bleu_score


def compute_weighted_ngram_bleu(
    reference: str,
    hypothesis: str,
    lang: str,
    max_n: int = 4,
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    """
    Compute the weighted n-gram match score, emphasizing important tokens.

    Args:
        reference: Reference code snippet
        hypothesis: Hypothesis code snippet
        lang: Language identifier
        max_n: Maximum n-gram order
        weights: Weights for n-grams from 1 to max_n

    Returns:
        Weighted BLEU score
    """
    if not reference or not hypothesis:
        raise ValueError("Reference and hypothesis cannot be empty")

    try:
        # Tokenize and get token weights
        ref_tokens = tokenize_code(reference, lang)
        hyp_tokens = tokenize_code(hypothesis, lang)

        ref_token_weights = get_token_weights(ref_tokens, lang)
        hyp_token_weights = get_token_weights(hyp_tokens, lang)

        if not ref_tokens or not hyp_tokens:
            return 0.0

        # Compute weighted n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = _get_ngrams_with_weights(ref_tokens, ref_token_weights, n)
            hyp_ngrams = _get_ngrams_with_weights(hyp_tokens, hyp_token_weights, n)

            if not hyp_ngrams:
                precisions.append(0.0)
                continue

            # Count weighted matches
            total_weight = 0.0
            matched_weight = 0.0

            ref_ngram_weights = {}
            for ngram, weight in ref_ngrams:
                if ngram not in ref_ngram_weights:
                    ref_ngram_weights[ngram] = 0.0
                ref_ngram_weights[ngram] += weight

            for ngram, weight in hyp_ngrams:
                total_weight += weight
                if ngram in ref_ngram_weights:
                    matched_weight += min(weight, ref_ngram_weights[ngram])
                    ref_ngram_weights[ngram] -= weight
                    if ref_ngram_weights[ngram] <= 0:
                        del ref_ngram_weights[ngram]

            precision = matched_weight / total_weight if total_weight > 0 else 0.0
            precisions.append(max(precision, 1e-9))

        # Compute brevity penalty (using token counts)
        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)

        if hyp_len > ref_len:
            brevity_penalty = 1.0
        else:
            brevity_penalty = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

        # Compute weighted BLEU score
        if all(p > 0 for p in precisions):
            log_precision_sum = sum(
                w * math.log(p) for w, p in zip(weights, precisions)
            )
            bleu_score = brevity_penalty * math.exp(log_precision_sum)
        else:
            bleu_score = 0.0

        return bleu_score

    except Exception as e:
        raise RuntimeError(f"Failed to compute weighted n-gram BLEU: {e}")


def compute_ast_match(reference: str, hypothesis: str, lang: str) -> float:
    """
    Compute the syntactic match score by comparing AST node sequences.

    Args:
        reference: Reference code snippet
        hypothesis: Hypothesis code snippet
        lang: Language identifier

    Returns:
        Syntactic match score based on AST node similarity
    """
    if not reference or not hypothesis:
        raise ValueError("Reference and hypothesis cannot be empty")

    try:
        parser = ts_manager.get_parser(lang)
        if not parser:
            raise RuntimeError(f"Parser not available for language: {lang}")

        # Parse both snippets
        ref_tree = parser.parse(reference.encode("utf-8"))
        hyp_tree = parser.parse(hypothesis.encode("utf-8"))

        # Extract AST node sequences (preorder traversal)
        ref_nodes = _extract_ast_nodes(ref_tree.root_node)
        hyp_nodes = _extract_ast_nodes(hyp_tree.root_node)

        if not ref_nodes or not hyp_nodes:
            return 0.0

        # Compute longest common subsequence
        lcs_length = _compute_lcs_length(ref_nodes, hyp_nodes)

        # Calculate match score
        total_nodes = len(ref_nodes) + len(hyp_nodes)
        if total_nodes == 0:
            return 0.0

        ast_score = (2.0 * lcs_length) / total_nodes
        return min(1.0, ast_score)

    except Exception as e:
        raise RuntimeError(f"Failed to compute AST match: {e}")


def tokenize_code(code: str, lang: str) -> List[str]:
    """
    Tokenize a code snippet into a list of tokens.

    Args:
        code: Code snippet to tokenize
        lang: Language identifier

    Returns:
        List of tokens extracted from the code
    """
    if not code:
        raise ValueError("Code cannot be empty")

    parser = ts_manager.get_parser(lang)
    if not parser:
        raise ValueError(f"No parser available for language: {lang}")

    tree = parser.parse(code.encode("utf-8"))
    tokens = []

    def _extract_tokens(node: Node):
        if node.child_count == 0:  # Leaf node
            text = code[node.start_byte : node.end_byte]
            if text.strip():
                tokens.append(text)
        else:
            for child in node.children:
                _extract_tokens(child)

    _extract_tokens(tree.root_node)
    if not tokens:
        raise ValueError("No tokens extracted from code")

    return tokens


def get_token_weights(tokens: List[str], lang: str) -> List[float]:
    """
    Assign weights to tokens based on their type.

    Args:
        tokens: List of tokens from tokenize_code
        lang: Language identifier

    Returns:
        List of weights corresponding to each token
    """
    if not tokens:
        raise ValueError("Tokens cannot be empty")

    parser = ts_manager.get_parser(lang)
    if not parser:
        raise ValueError(f"No parser available for language: {lang}")

    # Create a dummy code snippet from tokens to parse
    code = " ".join(tokens)
    tree = parser.parse(code.encode("utf-8"))

    weights = []
    token_index = 0

    def _assign_weights(node: Node):
        nonlocal token_index
        if node.child_count == 0:  # Leaf node
            if token_index < len(tokens):
                weight = _get_node_weight(node)
                weights.append(weight)
                token_index += 1
        else:
            for child in node.children:
                _assign_weights(child)

    _assign_weights(tree.root_node)

    # Ensure we have weights for all tokens
    while len(weights) < len(tokens):
        weights.append(TOKEN_WEIGHTS["default"])

    return weights[: len(tokens)]
