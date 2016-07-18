

Node = {
    children = {},
    correct = -1,
    --left = 1000000,
    --result = 0.5,
    --size = 0,
    --name = "",
    addchild = function(self, node)
            self.left = math.min(self.left, node.left)
            position = #self.children + 1
            for i = #self.children,1,-1 do
                if node.left < self.children[i].left then position = i end
            end
            for i = position,#self.children do
                self.children[i+1] = self.children[i]
            end
            self.children[position] = node
            self.size = self.size + node.size
        end,
    printout = function(self, z)
            zz = z or 1
            for i = 1, zz do io.write("|") end
            io.write(self.name .. "   " .. self.output .. "  c=" .. self.correct .. "\n")
            for i = 1, #self.children do
                    self.children[i]:printout(zz+1)
            end
        end,
    new = function(self, name)
            o = {}
            setmetatable(o, self)
            self.__index = self
            o.name = name
            o.children = {}
            o.left = 1000000
            o.size = 0
            return o
        end,
    newleaf = function(self, i, word)
            o = {}
            setmetatable(o, self)
            self.__index = self
            o.left = i
            o.name = word
            o.size = 1
            o.children = {}
            o.input = t(word)
            return o
        end
}


--[[
function Node.empty(nwords, ntree)
    Node.children = {}
    Node.left = {}
    Node.size = {}
    for i = 1, nwords do 
        Node.left[i] = i 
        Node.size[i] = 1 
    end
    for i = nwords+1, ntree do 
        Node.left[i] = 10000000 --infinity
        Node.size[i] = 0
    end
end

function Node.parent(child, parent)
    Node.size[parent] = Node.size[parent] + Node.size[child]
    Node.left[parent] = math.min(Node.left[parent], Node.left[child])
    Node.children[i] = Node.left[i]
end
]]


return Node
