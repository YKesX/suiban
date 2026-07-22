"""Bundled public-domain text sample for `suiban bench kv`.

Contents (all long out of copyright, reproduced verbatim):
- Abraham Lincoln, the Gettysburg Address (1863, Bliss copy) — US public domain.
- William Shakespeare, Sonnets 18 and 116 (1609) — public domain.

Used as the WORD POOL for the deterministic shuffled llama-perplexity corpus
(bench/kv.py build_ppl_corpus): real English tokens, deliberately shuffled so the
text cannot be memorized in-context — absolute PPL on it is meaningless by
construction, only the deltas between V-cache configs carry signal, and the
report says so.
"""

from __future__ import annotations

GETTYSBURG_ADDRESS = """\
Four score and seven years ago our fathers brought forth on this continent, a new
nation, conceived in Liberty, and dedicated to the proposition that all men are
created equal.

Now we are engaged in a great civil war, testing whether that nation, or any nation
so conceived and so dedicated, can long endure. We are met on a great battle-field of
that war. We have come to dedicate a portion of that field, as a final resting place
for those who here gave their lives that that nation might live. It is altogether
fitting and proper that we should do this.

But, in a larger sense, we can not dedicate — we can not consecrate — we can not
hallow — this ground. The brave men, living and dead, who struggled here, have
consecrated it, far above our poor power to add or detract. The world will little
note, nor long remember what we say here, but it can never forget what they did here.
It is for us the living, rather, to be dedicated here to the unfinished work which
they who fought here have thus far so nobly advanced. It is rather for us to be here
dedicated to the great task remaining before us — that from these honored dead we
take increased devotion to that cause for which they gave the last full measure of
devotion — that we here highly resolve that these dead shall not have died in vain —
that this nation, under God, shall have a new birth of freedom — and that government
of the people, by the people, for the people, shall not perish from the earth.
"""

SONNET_18 = """\
Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade
Nor lose possession of that fair thou owest;
Nor shall Death brag thou wander'st in his shade,
When in eternal lines to time thou growest:
So long as men can breathe or eyes can see,
So long lives this and this gives life to thee.
"""

SONNET_116 = """\
Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove:
O no; it is an ever-fixed mark,
That looks on tempests, and is never shaken;
It is the star to every wandering bark,
Whose worth's unknown, although his height be taken.
Love's not Time's fool, though rosy lips and cheeks
Within his bending sickle's compass come;
Love alters not with his brief hours and weeks,
But bears it out even to the edge of doom.
If this be error and upon me proved,
I never writ, nor no man ever loved.
"""

SAMPLE_TEXT = "\n\n".join((GETTYSBURG_ADDRESS, SONNET_18, SONNET_116))
