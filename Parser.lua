require 'ffi'


local Parser = {}

jvm = ffi.load('jvm.dll');
p1 = ffi.load('parser.dll')

function Parser.parse(str)
	--cmd = "java -cp \"../stanford-parser-full-2015-12-09/*;./java\" Parser \"" .. str .. "\" >result.txt "
	line = p1.parse(str)
	os.execute(cmd)
	local file = io.open("result.txt","r")
	io.input(file)
	line = io.read()
	line = io.read()
	return line
	--print(line)
end




return Parser
