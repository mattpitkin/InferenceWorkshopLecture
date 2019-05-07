# LaTeX compiler
#TEXC := pdflatex
TEXC := xelatex

default: pdf

pdf: lecture_notes.md
	echo "Convert Markdown text file to TeX file"
	multimarkdown -t beamer lecture_notes.md > lecture_notes.tex
	echo "Comile TeX file with xelatex"
	$(TEXC) lecture_notes.tex
	$(TEXC) lecture_notes.tex

clean:
	rm -f *.out *.aux *.log *.toc *.snm *.nav *.backup
