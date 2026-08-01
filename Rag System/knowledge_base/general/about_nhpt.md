---
doc_type: about
sources:
  - name: "Coursework scenario description"
    license: "Original"
---

# About the National Heritage Preservation Trust

## Remit

The National Heritage Preservation Trust (NHPT) manages historic sites
across the UK, with a dual mission: preserving the physical fabric of
significant historic buildings, and helping the public understand and
engage with the architectural and historical significance of what they
are visiting.

## The Challenges This System Addresses

NHPT currently faces two operational challenges that this prototype AI
system is designed to help with:

**Structural deterioration monitoring** has historically relied on manual
inspection, which is slow and produces inconsistent results between
different inspectors and sites. A computer vision component that can
assist in classifying and flagging building features is intended to
support, not replace, professional structural assessment.

**Visitor knowledge support** has been limited by the difficulty of
providing personalised, on-demand explanations of historical features at
scale. Most visitor information is currently static — printed panels or
guidebooks — and cannot answer a specific visitor's follow-up questions
about a particular building or feature in front of them.

## This Prototype

This system combines a computer vision classifier, trained to recognise
eight architectural styles represented across NHPT's UK site portfolio,
with a conversational assistant built on Retrieval-Augmented Generation
(RAG). A visitor can photograph a building feature, receive an automatic
style classification with a confidence score, and then ask natural-
language follow-up questions, which the assistant answers by retrieving
relevant material from NHPT's curated knowledge base rather than
generating answers from general knowledge alone. This grounding in
retrieved documents is intended to reduce the risk of the system
inventing incorrect historical claims, and every answer can cite the
specific document it was drawn from.

## Scope and Limitations

This is an educational prototype, not a deployed production system. It
does not perform structural safety assessment, does not replace expert
architectural or conservation advice, and its knowledge base — eight
architectural styles with supporting reference material — is
representative rather than exhaustive of the full range of historic
building styles found across NHPT's actual site portfolio.
