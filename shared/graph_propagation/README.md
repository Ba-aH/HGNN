| Key   | Traversal     | Semantic meaning                                             | Useful for LCR? |
|-------|---------------|--------------------------------------------------------------|-----------------|
| P     | —             | Raw paper feature (abstract or context-mean)                 | Yes — always included as self-feature |
| PP    | P→P           | Direct citation neighbours mean                              | Yes — papers you cite tend to be relevant |
| PPP   | P→P→P         | 2-hop citation chain                                         | Yes — captures broader citation neighbourhood |
| PC    | P→C           | Mean of context passages written by this paper               | Marginal — describes how this paper cites others |
| PCP   | P←C→P         | Papers co-cited in the same passage as this paper            | Yes — strongest signal |
| PCrP  | P→C←P         | Papers that share the same citing source paper               | Yes — bibliographic coupling via context |