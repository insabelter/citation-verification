# citation-verification

Inspired by Tianmai M. Zhang, Neil F. Abernethy: "Detecting Reference Errors in Scientific Literature with Large Language Models" (https://arxiv.org/abs/2411.06101)

Ablauf der Implementierung:
- vorgegebene Daten analysieren (Diagramme erstellen)
- Versuch, PDFs automatisch zu downloaden mit Crawler - nicht erfolgreich
- Automatisch in Zotero importieren über Titel und DOI unter Verwendung von https://anystyle.io/
- Auto-Download der PDFs über Zotero, falls nicht gefunden manuell überprüfen
    - Für einige PDF kein Institutional Access (dann über Google, sci-hub gesucht)
    - Einige PDFs sind retracted 
- TEI extraction using GROBID model (small vs. full model comparison)
    - improving the extraction by getting only the body text from the TEI (references for example are not useful for substaniating the statements)
    - Additionally: Extracting the text manually for some special papers, where the text extraction failed: r071, r147

Ideas for result improvements:
- use full GROBID model
- alayze and refactor extraction output
- use output_format variable for model prompts

Dataset Creation from DrInventor Corpus:
- Download DrInventor compiled Corpus (with extra annotations as original was not available)
- Analyze txt files to find the 5 papers with the least references (that use IEEE citation style for easy matching of citations to references)
- Download the citation papers and referenced reference papers (except websites or whole books)
- Citations were manually extracted from the papers
    - citing paper 02: mix of ^ and [] citations