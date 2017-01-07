

nullwrd = "[-]"


function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module, 'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

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

function nullVector(nbr)
	local w = {}
	for j = 1, nbr do
		w[j] = 0
	end
	return w
end

function v(word) -- vector representation
	local vec = dictionary[word:lower()]
	if word == nullwrd then return nullVector(vectorSize) end
	if vec == nil then
		--vec = Glove.searchmore(word:lower())
        --vec = nullVector(vectorSize)
        vec = {}
        for i = 1,vectorSize do
            vec[i] = -1 + math.random()*2
        end
        dictionary[word:lower()] = vec
	end
	return vec
end  

function t(word) -- tensor representation
	return torch.Tensor({v(word)})
end

function u(list) -- tensor representation
	return torch.Tensor({list})
end

if classnumber == 5 then
    bucket = function (x)
        if x<0.2 then return 0 end
        if x<0.4 then return 1 end
        if x<0.6 then return 2 end
        if x<0.8 then return 3 end
	    return 4;
    end
else
    bucket = function (x)
        if x<0.4 then return 1 end
        if x<0.6 then return 0 end
	    return 2;
    end
end

function tablebucket(x)
    x = x[1]
    local max = 1;
    for i = 1, classnumber do if x[i] > x[max] then max = i end end
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

