
input_without_tex=artykul


once : $(input_without_tex).tex
	pdflatex $(input_without_tex).tex


full : $(input_without_tex).tex
	pdflatex $(input_without_tex).tex
	bibtex $(input_without_tex)
	pdflatex $(input_without_tex).tex
	pdflatex $(input_without_tex).tex


xelatex : $(input_without_tex).tex
	xelatex $(input_without_tex).tex
	bibtex $(input_without_tex)
	xelatex $(input_without_tex).tex
	xelatex $(input_without_tex).tex

clean : 
	rm -f *.aux *.log *.bbl *.blg *.out  

