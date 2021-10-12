# 클래스 및 함수 선언부
import random


class Node() :
    def __init__(self):
        self.data = None
        self.link = None

def printNodes(start) :
    current = start

    print(current.data, end= ' ')
    while current.link != None :
        current = current.link
        print(current.data, end=' ')

    print()

def insertNodes(find, insert):
    global memory, head, current, pre

    if head.data == find :
        node = Node()
        node.data = insert
        node.link = head
        head = node
        return

    current = head
    while current.link != None :
        pre = current
        current = current.link

def deleteNodes(delete) :
    global memory, head, current, pre

    if head.data == delete:
        current = head
        head = head.link
        del(current)
        return

    current = head
    while current != None :
        pre = current
        current = current.link

#전역 변수부
current, pre, head = None, None, None
dataArray = [random.randint(0, 1000) for a in range(10)]
memory = []

#메인 코드부

if __name__ == '__main__' :

    node = Node()
    node.data = dataArray[0]
    head = node
    memory.append(node)

    for data in dataArray[1:] : #0번째는 위에서
        pre = node
        node = Node()
        node.data = data
        pre.link = node
        memory.append(node)

    printNodes(head)