katok = ['다현', '정연', '쯔위', '사나', '지효']

def insert_data(position, friend):
    katok.append(None)
    klen = len(katok)

    for i in range(klen-1, position, -1 ) :
        katok[i] = klen[i-1]
        katok[i-1] = None

    katok[position] = friend

insert_data(2, '솔라')
print(katok)
