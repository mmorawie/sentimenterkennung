
--nullwrd = "[-]"

-- global functions --

function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end

function map(func, array)
  local new_array = {}
  for i,v in ipairs(array) do
    new_array[i] = func(v)
  end
  return new_array
end

local Service = {}

function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end

function nullVector(nbr)
	local w = {}
	for j = 1, nbr do
		w[j] = 0
	end
	return w
end

function v(word) -- vector representation
	local vec = Service.parent.dictionary[word:lower()]
	if vec == nil then
        vec = {}
        for i = 1, Service.parent.vector_size do
            vec[i] = -1 + math.random()*2
        end
        if Service.parent.dict_max_size == nil or parent.dict_max_size > Service.parent.dict_size then 
            Service.parent.dictionary[word:lower()] = vec 
            Service.parent.dict_size = Service.parent.dict_size + 1
        end
	end
	return vec
end  

function t(word) -- tensor representation
	return torch.Tensor({v(word)})
end

function u(list) -- tensor representation
	return torch.Tensor({list})
end

function tablebucket(x)
    x = x[1]
    local max = 1;
    for i = 1, x:size()[1] do if x[i] > x[max] then max = i end end
    return max;
end

function print_r ( t )  
    local print_r_cache={}
    local function sub_print_r(t,indent)
        if (print_r_cache[tostring(t)]) then
            print(indent.."*"..tostring(t))
        else
            print_r_cache[tostring(t)]=true
            if (type(t)=="table") then
                for pos,val in pairs(t) do
                    if (type(val)=="table") then
                        print(indent.."["..pos.."] => "..tostring(t).." {")
                        sub_print_r(val,indent..string.rep(" ",string.len(pos)+8))
                        print(indent..string.rep(" ",string.len(pos)+6).."}")
                    elseif (type(val)=="string") then
                        print(indent.."["..pos..'] => "'..val..'"')
                    else
                        print(indent.."["..pos.."] => "..tostring(val))
                    end
                end
            else
                print(indent..tostring(t))
            end
        end
    end
    if (type(t)=="table") then
        print(tostring(t).." {")
        sub_print_r(t,"  ")
        print("}")
    else
        sub_print_r(t,"  ")
    end
    print()
end


return Service
