"""szg — spade-zero-gradient.

Research question (SPADE, arXiv:2608.19197, Future Directions): can the
Environment Designer improve through in-context learning rather than gradient
updates? Here: frozen designer weights (Sonnet), frozen solver (Haiku),
designer improvement only via (a) accumulated in-context environment memory
and (b) ShinkaEvolve-style evolution of the designer's written strategy.
Fitness: floored hint-based regret, guarded by validity gates (PREREG.md D-02).
"""

__version__ = "0.1.0"
