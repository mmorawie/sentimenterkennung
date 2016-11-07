

Node = {
    children = {},
    --correct = -1,
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
    printout = function(self)
            printout(self, 1)
        end,
    printout = function(self, z)
            zz = z or 1
            for i = 1, zz do io.write("|") end
            io.write(self.name .. "   " .. tablebucket( self.output ) + 1 .. "\n")
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
            o.correct = -1;
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
            o.correct = -1;
            return o
        end,
    setup = function(words, tree)
            local nodes = {}
	        for j = #words + 1,#tree do 	
                nodes[j] = Node:new("" .. j) 
            end
	        for j = 1,#words do 
                nodes[j] = Node:newleaf(j, words[j])
                nodes[tree[j]]:addchild(nodes[j])
            end
	        for j = #words+1,#tree-1 do 	nodes[tree[j]]:addchild(nodes[j]) end
	        for j = #words+1, #tree do 		nodes[j].name = nodes[j].children[1].name .. " " .. nodes[j].children[2].name end
            for j = 1, #tree do 		
                nodes[j].output = -1	
                nodes[j].correct = results[ phrases[ nodes[j].name ]  ]
            end
            nodes.wsize = #words
            nodes.tsize = #tree
            return nodes
        end
    
}


return Node
