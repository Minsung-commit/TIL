#클래스 / 함수 선언부
import random


class Node() :
    def __init__(self):
        self.data = None
        self.link = None

def printNodes(start):
    current = start
    if current == None :
            return
    print(current.data, end= ' ')
    while current.link != None :
        current = current.link
        print(current.data, end= ' ')
    print()

## 전역 변수부
memory = []
head, current, pre = None, None, None
# dataArray = ['다현','쯔위','사나','모모','지효'] # DB, 크롤링.....
dataArray = [random.randint(1000,9999) for a in range(20)]
## 메인 코드부분

if __name__ == '__main__' : #main() 함수는 프로그램의 진입점

    node = Node()
    node.data = dataArray[0]
    head = node
    memory.append(node)

    for data in dataArray[1:] : #['쯔위','사나','모모','지효']
        pre = node
        node = Node()
        node.data = data
        pre.link = node
        memory.append(node)


    printNodes(head)
