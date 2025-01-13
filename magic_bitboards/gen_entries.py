
# the output, in priority order
runs = ["./run5.txt", "./run2_cpu.txt", "./run4.txt"]

b_board = [None] * 64
r_board = [None] * 64

for r in runs:
    with open(r, "r") as fp:
        txt = fp.read()
    
    padding = int(txt[txt.find('(')+1:txt.find(')')].split()[0])

    for l in txt.split('\nStarting search ')[1:]:
        index = int(l.split()[0])
        mask = int(l.split()[1][1:-1], 16)
        res = int(l.split(' = ')[1].split()[0], 16)

        if res == 0:
            continue

        item = (mask, padding, res)
        board = r_board
        if index >= 64:
            board = b_board
            index -= 64

        if board[index] is not None:
            continue
        
        board[index] = item

# now take it all and output the lines

print("/*Bishop Board*/")
for i in range(len(b_board)):
    mask, padding, res = b_board[i]
    print(f"    starter({hex(mask)}, {padding}, {hex(res)}),")

print("/*Rook Board*/")
for i in range(len(r_board)):
    mask, padding, res = r_board[i]
    print(f"    starter({hex(mask)}, {padding}, {hex(res)}),")
