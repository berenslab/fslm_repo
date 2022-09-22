from invoke import task
from pathlib import Path

basepath = "./"

open_cmd = "open"

fig_names = {
    "1": "paper/fig1",
    "2": "paper/fig2",
    "3": "paper/fig3",
    "4": "paper/fig4",
}

@task
def convertpngpdf(c, fig):
    _convertsvg2pdf(c, fig)
    _convertpdf2png(c, fig)


########################################################################################
# Helpers
########################################################################################
@task
def _convertsvg2pdf(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_svg2pdf(c, str(f + 1))
        return
    srcs = list(Path(f"{basepath}/{fig_names[fig]}/svg/").glob("*.svg"))
    dests = (f"{str(x).replace('/svg/','/fig/')[:-4]}.pdf" for x in srcs)
    for src, dest in zip(srcs, dests):
        c.run(f"inkscape {str(src)} --export-pdf={str(dest)}")


@task
def _convertpdf2png(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_pdf2png(c, str(f + 1))
        return
    pathlist = Path(f"{basepath}/{fig_names[fig]}/fig/").glob("*.pdf")
    for path in pathlist:
        c.run(
            f'inkscape {str(path)} --export-png={str(path)[:-4]}.png -b "white" --export-dpi=300'
        )