
local Glove = {}

function Glove.loadDictionary(path, verbose)
	--"./glove.6B/glove.6B.50d.txt"
	if verbose then print( "[ loading vector representation        ]") end
	local file = io.open(path, "r")
	io.input(file)
	local dico = {}
	local ii = 0; local jj = 0
	while true do
		ii = ii + 1
		if ii%10000 == 0 then jj = jj+1
			--if verbose then print( "...." .. tostring(jj) .. " \t\t" .. tostring(ii)) end
			if verbose then io.write("'") end
		end
		local line = io.read()
		if line == nil then break end
		local tab = line:split(" ")
		local w = {}
		for j = 1, #tab-1 do
			w[j] = tonumber(tab[j+1])
		end
		dico[tab[1]] = w -- torch.Tensor(w)
	end
	io.close(file)
	if verbose then io.write("\n") end
	return dico
end

function Glove.lookup(path)
	local file = io.open(path, "r")
	print( "< loading vector representation        >")
	io.input(file)
	local dico = {}
	local ii = 0;
	while true do
		ii = ii + 1
		if ii%10000 == 0 then io.write("'") end
		local line = io.read()
		if line == nil then break end
		local tab = line:split(" ")
		if dictionary[ tab[1] ] == -1 then
			local w = {}
			for j = 1, #tab-1 do
				w[j] = tonumber(tab[j+1])
			end
			dico[tab[1]] = w
		end
	end
	io.close(file)
	io.write("\n")
	return dico
end

return Glove
