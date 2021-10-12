katok = [] #리스트 생성

def add(name) : # 함수지정

    katok.append(None)
    klen = len(katok)
    katok[klen-1] = name

add('사나')
add('지효')
add('쯔위')

print(katok)