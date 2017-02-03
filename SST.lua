

local SST = {
	dir = './sst'
}

local binSST = {}

function SST.loadResults()
	local file = io.open( SST.dir .. '/sentiment_labels.txt', 'r')
	io.input(file)
	local results = {};
	local i = 1;
	io.read()
	while true do
		local line = io.read()
		if line == nil then break end
		line = line:split("|")
		results[i-1] = tonumber(line[2])
		i = i+1
	end
	io.close(file)
	return results
end

function SST.loadSplit()
	local file = io.open( SST.dir .. '/datasetSplit.txt','r')
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
	local file = io.open(SST.dir .. '/SOStr.txt','r')
	io.input(file)
	local sentences = io.read('*all'):split('\n')
	io.close(file)
	file = io.open(SST.dir .. '/datasetSentences.txt','r')
	io.input(file)
	text = {}; local i = 1
	while true do
		local line = io.read(); if line == nil then break end
		line = line:split('\t')
		text[i] = line[2]; i = i + 1
	end
	io.close(file)
	return sentences
end



function SST.loadPhrases()
	local file = io.open(SST.dir .. '/dictionary.txt','r')
	io.input(file)
	local phrases = {};
	while true do
		local line = io.read(); if line == nil then break end
		line = line:split('|')
		phrases[line[1]] = tonumber(line[2]);
	end
	io.close(file)
	phrases["!"] = 0
	return phrases
end

function SST.loadTrees()
	local file = io.open(SST.dir ..'/STree.txt','r'); io.input(file)
	local trees = {}; local i = 1;
	while true do
		local line = io.read(); if line == nil then break end
		line = line:split('|')
		trees[i] = map(tonumber,line); i = i+1
	end
	io.close(file)
	return trees
end

function SST.loadsets()
	SST.set 		= SST.loadSplit()
	SST.results 	= SST.loadResults()
	SST.sentences 	= SST.loadSentences()
	SST.phrases 	= SST.loadPhrases()
	SST.trees 		= SST.loadTrees()
	SST.trainset = {}; local a = 1;
	SST.testset = {}; local b = 1;
	SST.devset = {}; local c = 1;

	for i = 1, #SST.sentences do
		if SST.set[i] == "1" then SST.trainset[a] = i; a = a + 1 end
		if SST.set[i] == "2" then SST.testset[b] = i; b = b + 1 end
		if SST.set[i] == "3" then SST.devset[c] = i; c = c + 1 end
	end
	

end



return SST
