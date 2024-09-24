import requests

# #import embl_gems and get asscesion numbers
# with open("data/embl_gems/model_list.tsv") as f:
#     next(f)
#     lines = f.readlines()

# ids = [l.split("\t")[0] for l in lines]

#loop though assembly and get ftp
with open("data/embl_gems/ftpdirs.txt", 'w') as g:
    with open("data/assembly_summary.txt") as f:
        next(f)
        next(f)
        for l in f:
            genome = l.split("\t")[11]
            if genome == "Complete Genome":
                ftp_raw = l.split("\t")[19]
                ftp = ftp_raw + "/" + ftp_raw.split("/")[-1] + "_protein.faa.gz"
                g.write(ftp + "\n")

            

