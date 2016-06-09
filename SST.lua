

local SST = {}

function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end


function SST.loadResults()
	local file = io.open("./sst/sentiment_labels.txt","r")
	io.input(file)
	local results = {};
	local i = 1;
	io.read()
	while true do
		local line = io.read()
		if line == nil then break end
		line = line:split("|")
		results[i] = tonumber(line[2])
		i = i+1
	end
	io.close(file)
	return results
end

function SST.loadSplit()
	local file = io.open("./sst/datasetSplit.txt","r")
	io.input(file)
	local set = {}; local i = 1; io.read()
	while true do
		local line = io.read(); if line == nil then break end
		line = line:split(",")
		set[i] = line[2]; i = i + 1
	end
	io.close(file)
	return set
end

function SST.loadSentences()
	local file = io.open("./sst/SOStr.txt","r")
	io.input(file)
	local sentences = io.read("*all"):split("\n")
	io.close(file)
	--print(string.byte(sentences:split("\n")[1]:split("|")[1],2))
	file = io.open("./sst/datasetSentences.txt","r")
	io.input(file)
	text = {}; local i = 1
	while true do
		local line = io.read(); if line == nil then break end
		line = line:split("\t")
		text[i] = line[2]; i = i + 1
	end
	io.close(file)
	return sentences
end



function SST.loadPhrases()
	local file = io.open("./sst/dictionary.txt","r")
	io.input(file)
	local phrases = {};
	while true do
		local line = io.read(); if line == nil then break end
		line = line:split("|")
		phrases[line[1]] = tonumber(line[2]);
	end
	io.close(file)
	phrases["!"] = 0
	return phrases
end

function SST.loadTrees()
	local file = io.open("./sst/STree.txt","r"); io.input(file)
	local trees = {}; local i = 1;
	while true do
		local line = io.read(); if line == nil then break end
		line = line:split("|")
		trees[i] = map(tonumber,line); i = i+1
	end
	io.close(file)
	return trees
end



return SST
