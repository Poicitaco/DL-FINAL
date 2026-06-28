# Build LaTeX report
# Chay tren WSL: make

PDF = main.pdf
TEX = main.tex

all: $(PDF)

$(PDF): $(TEX) references.bib
	pdflatex $(TEX)
	bibtex main
	pdflatex $(TEX)
	pdflatex $(TEX)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk

view: $(PDF)
	evince $(PDF) &

.PHONY: all clean view
