anecs = list()
anecs_cnt = 0
anecs_len = []

with open("anec_bol.txt", encoding="utf-8") as file:
    textin = file.read().split("\n\n")
    for anec in textin:
        toadd = True
        anec_lined = anec.split("\n")
        if len(anec_lined) >= 2:
            anec_length = len(anec_lined)
            if not anec_lined[0][0] == "-":
            	for i in range(1, anec_length):
                    if not anec_lined[i].count("-") == 1 or not anec_lined[i][0] == "-":
                    	toadd = False
            else:
                toadd = False
        else:
            toadd = False

        if toadd:
            if len(anec) >= 10:
                anecs.append(anec)
                anecs_cnt += 1
                anecs_len.append(len(anec))

textout = "\n\n".join(anecs)


with open("anecs4.txt", mode="w", encoding="utf-8") as file:
    file.write(textout)

with open("anecs4.txt", mode="r", encoding="utf-8") as file:
    text = file.read()
    print(f"text:\n{text}")
    print(f"total length of text (chars count): {len(text)}")
    print(f"average anec len: {len(text)//anecs_cnt}")
    print(f"max anec len: {max(anecs_len)}")
    print(f"min anec len: {min(anecs_len)}")


"""
with open("anec_bol.txt", encoding="utf-8") as file:
    lines = file.readlines()
    stop = len(lines)
    i = 0
    templen = 0
    temp_cnt = 0
    sep = False
    cnt = 0
    outxt = list()

    while i < stop - 1:
        if lines[i] == "\n":
            if not sep:
                outxt.append(lines[i])
            sep = True
            i += 1
            if len(lines[i]) > 1:
                if not lines[i][0] == "-":
                    outxt.append(lines[i])
                    templen = 1
                    temp_cnt = lines[i].count("-")
                    if lines[i].count("-") > 0:
                        temp_cnt = 3
                    while i < stop - 1 and not lines[i+1]  == "\n":
                        outxt.append(lines[i+1])
                        temp_cnt += lines[i+1].count("-")
                        if lines[i+1].count("-") > 1 or not lines[i+1][0] == "-":
                            temp_cnt = 3
                        templen += 1
                        i += 1
                    if not templen == 3 or not temp_cnt == 2:
                        outxt = outxt[:len(outxt) - templen]
                    else:
                        sep = False
        i += 1

text = ""
anecs_len = []
anecs_cnt = 0
tmp_len = 0
for line in outxt:
    if line == "\n":
        anecs_cnt += 1
        anecs_len.append(tmp_len)
        tmp_len = 0
    else:
        tmp_len += len(line)
    text += line

anecs_cnt -= 1




with open("anecs3.txt", mode="w", encoding="utf-8") as file:
    file.write(text)

text = ""
with open("anecs3.txt", mode="r", encoding="utf-8") as file:
    text = file.read()
    print(f"text:\n{text}")
    print(f"total length of text (chars count): {len(text)}")
    print(f"average anec len: {len(text)//anecs_cnt}")
    print(f"max anec len: {max(anecs_len[1:])}")
    print(f"min anec len: {min(anecs_len[1:])}")
"""
