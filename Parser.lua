

local Parser = {}

local ffi = require 'ffi'
local started = false

ffi.cdef[[
char* parse(char* buf);
int test();
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

return Parser
