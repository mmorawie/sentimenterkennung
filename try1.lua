require 'torch'
require 'nn'
require 'nngraph'
--require 'rnn'

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

function loadDictionary()
	print( " loading vector representation ")
	local file = io.open("./glove.6B/glove.6B.50d.txt","r")
	io.input(file)
	local dico = {}
	local ii = 0; local jj = 0
	while true do
		ii = ii + 1
		if ii%10000 == 0 then jj = jj+1
			print( "...." .. tostring(jj) .. " \t\t" .. tostring(ii))
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
	return dico
end

function nullVector(nbr)
	local w = {}
	for j = 1, nbr do
		w[j] = 0
	end
	return w
end

function loadResults()
	local file = io.open("./sst/sentiment_labels.txt","r")
	io.input(file)
	results = {}; local i = 1; io.read()
	while true do
		local line = io.read()
		if line == nil then break end
		line = line:split("|")
		results[i] = tonumber(line[2])
		i = i+1
	end
	io.close(file)
end

function loadSplit()
	local file = io.open("./sst/datasetSplit.txt","r")
	io.input(file)
	set = {}; local i = 1; io.read()
	while true do
		local line = io.read(); if line == nil then break end
		line = line:split(",")
		set[i] = line[2]; i = i + 1
	end
	io.close(file)
end

function loadSentences()
	local file = io.open("./sst/SOStr.txt","r")
	io.input(file)
	sentences = io.read("*all"):split("\n")
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
end

function loadPhrases()
	local file = io.open("./sst/dictionary.txt","r")
	io.input(file)
	phrases = {};
	while true do
		local line = io.read(); if line == nil then break end
		line = line:split("|")
		phrases[line[1]] = tonumber(line[2]);
	end
	io.close(file)
	phrases["!"] = 0
end

function loadTrees()
	local file = io.open("./sst/STree.txt","r"); io.input(file)
	trees = {}; local i = 1;
	while true do
		local line = io.read(); if line == nil then break end
		line = line:split("|")
		trees[i] = map(tonumber,line); i = i+1
	end
	io.close(file)
end

function v(word) -- vector representation
	local vec = dictionary[word:lower()]
	if vec == nil then
		vec = nullVector(vectorSize) --vector size
	end
	return vec
end

function t(word) -- tensor representation
	return torch.Tensor({v(word)})
end

function u(list) -- tensor representation
	return torch.Tensor({list})
end

print("Start")

loadSplit()
loadResults()
loadSentences()
loadPhrases()
loadTrees()

dictionary = loadDictionary()
vectorSize = #(dictionary["do"]) -- vector size

GRU = require 'GRU'

function getcorrectvalues(words, tree)
	local correct = {}
	local priority = {}
	for j = 1,#words do priority[j] = j end
	for j = 1,#tree do

		if priority[j] == nil then priority[j] = j end
		--print("\t\t",words[j], priority[j])
		if results[phrases[words[j]]] == nil then results[phrases[words[j]]] = 0 end
		correct[j] = u({results[phrases[words[j]]]})
		if words[tree[j]] == nil then
			words[tree[j]] = words[j]
			priority[tree[j]] = priority[j]
		else
			if priority[tree[j]] < priority[j] then
				words[tree[j]] = words[tree[j]] .. " " .. words[j]
			else
				words[tree[j]] = words[j] .." " .. words[tree[j]]; priority[tree[j]] = priority[j]
			end
		end
	end
	return correct
end

function train(words, tree, netwrk)
	local output = {}
	local input = {}
	local correct = getcorrectvalues(words, tree)
	local layer = {}
	for j = 1,#words do input[j]= {t(words[j]), u({0})} end
	for j = #words + 1,#tree do input[j]= {t("[-]"), u({0})} end
	input[0]= {"", u({0})}
	criterion = nn.MSECriterion()
	-- fwd
	tot = 0;
	for j = 1,#tree do
		layer[j] = netwrk:clone('weight', 'bias')
		layer[j]:forward(input[j])
		a = criterion:forward(layer[j].output[1], correct[j])
		tot = tot + a;
		input[tree[j]][2] = torch.add( input[tree[j]][2] , layer[j].output[1] )
		layer[j]:zeroGradParameters()
	end
	print(" -- >", tot/#tree)
	-- bwd
	for j = #tree,1,-1 do
		b = criterion:backward( layer[j].output, correct[j] )
		layer[j]:backward(input[j], b)
		layer[j]:updateParameters(0.001)
	end
	newntwk = GRU.create(50, 1)
	for j = 1,#tree do
		newntwk = layer[j]:clone()
	end
	return newntwk
end

function bucket(x)
	if x<0.2 then return 0 end
	if x<0.4 then return 1 end
	if x<0.6 then return 2 end
	if x<0.8 then return 3 end
	return 4;
end

totalcorrect = 0;
totalerror = 0;
total = 0;
sentcorrect = 0;
senttotal = 0;
sentme = 0;

function test(words, tree, netwrk)
	local output = {}
	local input = {}
	local correct = getcorrectvalues(words, tree)
	local layer = {}
	for j = 1,#words do input[j]= {t(words[j]), u({0})} end
	for j = #words + 1,#tree do input[j]= {t("[-]"), u({0})} end
	input[0]= {"", u({0})}
	local diff = 0
	for j = 1,#tree do
		netwrk:forward(input[j])
		diff = correct[j][1][1] -netwrk.output[1][1]
		if bucket(correct[j][1][1]) == bucket(netwrk.output[1][1]) then totalcorrect = totalcorrect + 1 end
		totalerror = totalerror + math.abs(diff);
		total = total + 1;
		print("\t\t", (correct[j][1][1] - netwrk.output[1][1] ) )
		input[tree[j]][2] = torch.add( input[tree[j]][2] , netwrk.output[1] )
	end
	if bucket(correct[#tree][1][1]) == bucket(netwrk.output[1][1]) then sentcorrect = sentcorrect + 1 end
	senttotal = senttotal + 1
	sentme = sentme + math.abs(diff)
end


network =  GRU.create(50, 1)
for i = 1, 900 do
	local words = sentences[i]:split("|")
	local tree = trees[i]
	if set[i] == "1" then
		network = train(words,tree, network)
	end
end
torch.save("./ntwk2", network)



for i = 1, 1000 do
	local words = sentences[i]:split("|")
	local tree = trees[i]
	if set[i] == "2" then
		test(words,tree, network)
	end
end
print(" fraction correct  : \t", totalcorrect/total)
print(" mean error        : \t", totalerror/total)
print(" sentences correct : \t", sentcorrect/senttotal)
print(" sentences mean err: \t", sentme/senttotal)
