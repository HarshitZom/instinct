from typing import List, Tuple, Dict, Set, Optional, Any
import re
try:
    import tree_sitter
    from tree_sitter import Language, Parser, Node
except ImportError:
    raise ImportError("tree-sitter is required. Install with: pip install tree-sitter")

# Language mappings
LANGUAGE_EXTENSIONS = {
    'py': 'python',
    'python': 'python',
    'java': 'java',
    'js': 'javascript',
    'javascript': 'javascript',
    'ts': 'typescript',
    'typescript': 'typescript',
    'cpp': 'cpp',
    'c++': 'cpp',
    'c': 'c',
    'h': 'c',
    'hpp': 'cpp',
    'rs': 'rust',
    'rust': 'rust'
}

# Token type weights for weighted n-gram matching
TOKEN_WEIGHTS = {
    'keyword': 1.0,
    'identifier': 0.8,
    'type': 0.8,
    'function': 0.9,
    'class': 0.9,
    'number': 0.5,
    'string': 0.5,
    'comment': 0.1,
    'operator': 0.6,
    'punctuation': 0.2,
    'default': 0.2
}

def _simple_tokenize(code: str) -> List[str]:
    """Simple tokenization using regex patterns."""
    # Pattern to match identifiers, numbers, operators, and punctuation
    pattern = r'\b\w+\b|[^\w\s]|\n'
    tokens = re.findall(pattern, code)
    return [token for token in tokens if token.strip() and token != '\n']

def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a list of tokens."""
    if n <= 0 or n > len(tokens):
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def _get_ngrams_with_weights(tokens: List[str], weights: List[float], n: int) -> List[Tuple[Tuple[str, ...], float]]:
    """Extract n-grams with their average weights."""
    if n <= 0 or n > len(tokens):
        return []
    
    ngrams_with_weights = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        avg_weight = sum(weights[i:i+n]) / n
        ngrams_with_weights.append((ngram, avg_weight))
    
    return ngrams_with_weights

def _extract_ast_nodes(node: Node) -> List[str]:
    """Extract AST node types in preorder traversal."""
    nodes = [node.type]
    for child in node.children:
        nodes.extend(_extract_ast_nodes(child))
    return nodes

def _compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute the length of the longest common subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def _tree_to_variable_index(node: Node, index_to_code: Dict) -> List[Tuple[int, int]]:
    """Extract variable indices from a tree node."""
    if (
        len(node.children) == 0 or node.type in ["string_literal", "string", "character_literal"]
    ) and node.type != "comment":
        index = (node.start_point, node.end_point)
        if index in index_to_code:
            _, code = index_to_code[index]
            if node.type != code:
                return [(node.start_point, node.end_point)]
            else:
                return []
        else:
            return []
    else:
        code_tokens = []
        for child in node.children:
            code_tokens += _tree_to_variable_index(child, index_to_code)
        return code_tokens


def _get_identifier_name(node: Node) -> Optional[str]:
    """Extract identifier name from a node."""
    if node.type == 'identifier':
        return node.text.decode('utf-8') if node.text else None
    return None

def _get_node_weight(node: Node) -> float:
    """Get weight for a Tree-sitter node based on its type."""
    node_type = node.type
    
    # Language keywords and important constructs
    if node_type in ['def', 'class', 'function', 'if', 'else', 'for', 'while', 'return', 'import', 'from']:
        return TOKEN_WEIGHTS['keyword']
    elif node_type in ['identifier', 'type_identifier']:
        return TOKEN_WEIGHTS['identifier']
    elif node_type in ['number', 'integer', 'float']:
        return TOKEN_WEIGHTS['number']
    elif node_type in ['string', 'string_literal']:
        return TOKEN_WEIGHTS['string']
    elif node_type == 'comment':
        return TOKEN_WEIGHTS['comment']
    elif node_type in ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=']:
        return TOKEN_WEIGHTS['operator']
    else:
        return TOKEN_WEIGHTS['default']

def _get_heuristic_weight(token: str, lang: str) -> float:
    """Get weight for a token using heuristic rules."""
    # Language-specific keywords
    keywords = {
        'python': {'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'import', 'from', 'as', 'return', 'yield', 'break', 'continue', 'pass', 'and', 'or', 'not', 'in', 'is', 'lambda', 'global', 'nonlocal'},
        'java': {'public', 'private', 'protected', 'static', 'final', 'abstract', 'class', 'interface', 'extends', 'implements', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'try', 'catch', 'finally', 'throw', 'throws', 'return', 'break', 'continue', 'new', 'this', 'super', 'null', 'true', 'false'},
        'javascript': {'var', 'let', 'const', 'function', 'class', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'try', 'catch', 'finally', 'throw', 'return', 'break', 'continue', 'new', 'this', 'null', 'undefined', 'true', 'false'},
        'typescript': {'var', 'let', 'const', 'function', 'class', 'interface', 'type', 'enum', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'try', 'catch', 'finally', 'throw', 'return', 'break', 'continue', 'new', 'this', 'null', 'undefined', 'true', 'false'},
        'c': {'int', 'char', 'float', 'double', 'void', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'break', 'continue', 'return', 'struct', 'union', 'enum', 'typedef', 'static', 'extern', 'const', 'volatile'},
        'cpp': {'int', 'char', 'float', 'double', 'void', 'bool', 'class', 'struct', 'namespace', 'using', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'try', 'catch', 'throw', 'return', 'break', 'continue', 'new', 'delete', 'this', 'public', 'private', 'protected', 'virtual', 'static', 'const', 'template'},
        'rust': {'fn', 'struct', 'enum', 'impl', 'trait', 'mod', 'use', 'pub', 'let', 'mut', 'const', 'static', 'if', 'else', 'match', 'for', 'while', 'loop', 'break', 'continue', 'return', 'where', 'move', 'ref', 'self', 'Self', 'super', 'crate', 'async', 'await', 'unsafe', 'extern', 'type', 'macro_rules'}
    }
    
    lang_keywords = keywords.get(LANGUAGE_EXTENSIONS.get(lang, lang), set())
    
    if token in lang_keywords:
        return TOKEN_WEIGHTS['keyword']
    elif token.isidentifier():
        return TOKEN_WEIGHTS['identifier']
    elif token.isdigit() or re.match(r'^\d*\.?\d+$', token):
        return TOKEN_WEIGHTS['number']
    elif token.startswith('"') or token.startswith("'"):
        return TOKEN_WEIGHTS['string']
    elif token in '+-*/=<>!&|':
        return TOKEN_WEIGHTS['operator']
    else:
        return TOKEN_WEIGHTS['default']