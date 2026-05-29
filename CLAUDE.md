# CLAUDE.md — Thesis Paper Writing Rules

## Template Structure

- **Class**: `memoir` (a4paper, 12pt, oneside, openbib, oldfontcommands)
- **Citation style**: `unsrt` (numbered in order of appearance, IEEE-like)
- **Packages**: `natbib` with `[square,numbers,sort&compress]`, `hyperref`, `graphicx`, `amsmath`, `amssymb`, `subcaption`, `pgf`
- **Spacing**: `\OnehalfSpacing`
- **Numbering depth**: `\setsecnumdepth{subsection}`

## File Organization

```
paper/
  memoirthesis.tex       (root — imports everything)
  thesisbiblio.bib       (bibliography, unsrt style)
  frontmatter/
    title.tex            (title page)
    abstract.tex         (structured abstract)
    dedication.tex       (optional)
    declaration.tex      (optional)
  chapters/
    chapter01/
      introduction.tex   (Introduction)
    chapter02/
      main.tex           (Related Work + Methodology + Experiments + Results)
      section2-1.tex     (Related Work subsections)
      fig02/             (figures)
    chapter03/
      conclusion.tex     (Conclusion)
      table03/           (tables)
    appendices/
      app0A.tex          (Appendix)
```

## Citation Rules

- Use `\cite{key}` for single citations.
- Use `\cite{key1,key2}` for multiple citations.
- Bibliography style: `\bibliographystyle{unsrt}`.
- All uncertain citations MUST use `[CITE-PLACEHOLDER: description]` in the text, NOT invented bib entries.
- Only include bib entries for VERIFIED papers (landmark works, papers I can confirm exist).

## Figure Rules

- Use `\begin{figure}[H]` with `\centering`.
- Subfigures use `subcaption` package: `\begin{subfigure}{0.3\textwidth}`.
- Label format: `\label{Fig.2.1}` for main figure, `\label{Fig.2.1.a}` for subfigure.
- All figures MUST be referenced in text before they appear.
- Save figures to `paper/chapters/chapter02/fig02/`.

## Table Rules

- Use standard `\begin{table}[H]` with `\centering`.
- Label format: `\label{Tab.3.1}`.
- All tables MUST be referenced in text.

## Mathematical Notation

- Scalars: lowercase italic ($x$, $\lambda$)
- Vectors: lowercase bold ($\mathbf{x}$)
- Matrices: uppercase bold ($\mathbf{W}$)
- Number sets: blackboard bold ($\mathbb{R}$)
- All referenced equations must be numbered: `\begin{equation}`

## Domain Consistency

- ALWAYS refer to the domain as **Astana bus network** (NOT Almaty Metro).
- The data is synthetic, generated from OSM with realistic simulation.
- The model is DTS-GSSF (Dual-Timescale Graph Gated Forecasting).

## Voice and Tense

- Past tense for experiments and results.
- Present tense for general facts, paper structure, and contributions.
- Use "we" throughout (never "I").
- Active voice preferred.

## Forbidden Phrases

- "In this paper, we propose a novel..." → be direct and specific.
- "State-of-the-art results" → qualify with benchmark and year.
- "Obviously" / "Clearly" / "Trivially" → never use.
- "The proposed method" → give the method a name and use it.
