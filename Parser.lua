

local Parser = {}

local ffi = require 'ffi'
local started = false

ffi.cdef[[
// parsing
char* parse(char* buf);
int test();
// vizualizartion
void init(int n);
void setLeaf(char* str, int sent, int i);
void setNode(char* str, int sent, int i, int j, int k);
void display();
// starting a JVM
int startJVM();
]]

local jvm = ffi.load('jvm.dll');
local parser = ffi.load('parser.dll')

function Parser.start()
	ret = parser.startJVM()
	if ret == 0 then 
		started = true
	else 
		started = false
	end
	return ret
end

function Parser.parse(str)
	strl = string.len(str)
	local buf = ffi.new("char[?]", #str)
	ffi.copy(buf, str)
	if started == false then
		Parser.start()
	end
	if started == false then
		return "err"
	end
	zz = parser.parse(buf)
	return ffi.string(zz)
end


function Parser.drawTree(nodes)
	if started == false then
		Parser.start()
	end
	if started == false then
		return "err"
	end
	parser.init(#nodes)
	for i = 1, #nodes do
		local str = nodes[i].name
		local buf = ffi.new("char[?]", #str)
		ffi.copy(buf, str)
		local sent = 1-- bucket(nodes[i].correct)
		if #nodes[i].children == 0 then
			parser.setLeaf(buf, sent, i)
		else
			parser.setNode(buf, sent, i, nodes[i].children[1].index, nodes[i].children[2].index)
		end
	end
	parser.display()
end

return Parser
