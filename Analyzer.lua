require 'torch'
require 'nn'
require 'nngraph'
GRU = require 'GRU2'

local Analyzer = {}

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

function load()
	SST 		= require 'SST'
	Glove		= require 'Glove'
	set 		= SST.loadSplit()
	results 	= SST.loadResults()
	sentences 	= SST.loadSentences()
	phrases 	= SST.loadPhrases()
	trees 		= SST.loadTrees()

	dictionary = Glove.loadDictionary('./glove.6B/glove.6B.50d.txt',true)
	vectorSize = #(dictionary["do"]) -- vector size
end

function dload()
	SST 		= require 'SST'
	Glove		= require 'Glove'
	set 		= SST.loadSplit()
	results 	= SST.loadResults()
	sentences 	= SST.loadSentences()
	phrases 	= SST.loadPhrases()
	trees 		= SST.loadTrees()

	dictionary = {}
	vectorSize = 300
end

function getcorrectvalues(words, tree)
	local correct = {}
	local priority = {}
	for j = 1,#words do priority[j] = j end
	for j = 1,#tree do

		if priority[j] == nil then priority[j] = j end
		--print("\t\t",words[j], priority[j])
		if results[phrases[words[j]]] == nil then results[phrases[words[j]]] = 0 end
		correct[j] = u({results[phrases[words[j]]],results[phrases[words[j]]]})
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

function train(words, tree, netwrk, uc)
	local output = {}
	local input = {}
	local min = {}
	local correct = getcorrectvalues(words, tree)
	local layer = {}
	for j = 1,#words do input[j]= {t(words[j]), u({0,0})} end
	for j = #words + 1,#tree do input[j]= {t("[-]"), u({0,0})} end
	for j = 1,#words do min[j] = j end
	for j = #words + 1, #tree do min[j] = 1000000 end
	min[0] = 1000000
	input[0]= {"", u({0,0})}
	criterion = nn.MSECriterion()
	-- fwd
	tot = 0;
	for j = 1,#tree do
		--print("-- " .. j .. " " .. min[j], input[tree[j]][2])
		layer[j] = netwrk:clone('weight', 'bias')
		layer[j]:forward(input[j])
		a = criterion:forward(layer[j].output[1], correct[j])
		tot = tot + a;
		if min[j] < min[tree[j]] then
			min[tree[j]] = min[j]
			input[tree[j]][2][1][2] = input[tree[j]][2][1][1]
			input[tree[j]][2][1][1] = layer[j].output[1][1]
		else
			input[tree[j]][2][1][1] = layer[j].output[1][1]
		end
		layer[j]:zeroGradParameters()
	end
	print(tot/#tree,", ")
	-- bwd
	newntwk = layer[#tree]:clone('weight','bias')
	for j = #tree,1,-1 do
		b = criterion:backward( layer[j].output[1], correct[j] )
		layer[j]:backward(input[j], b)
		layer[j]:updateParameters(uc)
		--newntwk:backward(input[j], b)
		--newntwk:zeroGradParameters()
		newntwk:updateParameters(uc)
	end
	newntwk = layer[#tree]:clone('weight','bias')
	return newntwk
end


function train2(words, tree, netwrk, uc)
	local output = {}
	local input = {}
	local min = {}
	local correct = getcorrectvalues(words, tree)
	local layer = {}
	for j = 1,#words do input[j]= {t(words[j]), u({0,0})} end
	for j = #words + 1,#tree do input[j]= {t("[-]"), u({0,0})} end
	for j = 1,#words do min[j] = j end
	for j = #words + 1, #tree do min[j] = 1000000 end
	min[0] = 1000000
	input[0]= {"", u({0,0})}
	criterion = nn.MSECriterion()
	tot = 0;
	for j = 1,#tree do
		--print("-- " .. j .. " " .. min[j], input[tree[j]][2])
		newntwk = netwrk:clone('weight', 'bias')
		newntwk:forward(input[j])
		a = criterion:forward(newntwk.output[1], correct[j])
		tot = tot + a;
		if min[j] < min[tree[j]] then
			min[tree[j]] = min[j]
			input[tree[j]][2][1][2] = input[tree[j]][2][1][1]
			input[tree[j]][2][1][1] = newntwk.output[1][1]
		else
			input[tree[j]][2][1][1] = newntwk.output[1][1]
		end
		newntwk:zeroGradParameters()
		b = criterion:backward( newntwk.output, correct[j] )
		newntwk:backward(input[j], b)
		newntwk:updateParameters(uc)
	end
	print(" -- >", tot/#tree)
	return newntwk
end


function bucket(x)
	if x<0.2 then return 0 end
	if x<0.4 then return 1 end
	if x<0.6 then return 2 end
	if x<0.8 then return 3 end
	return 4;
end

function bucket2(x)
	if x<0.33 then return 0 end
	if x<0.66 then return 1 end
	return 2;
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
	local min = {}
	local correct = getcorrectvalues(words, tree)
	local layer = {}
	for j = 1,#words do input[j]= {t(words[j]), u({0,0})} end
	for j = #words + 1,#tree do input[j]= {t("[-]"), u({0,0})} end
	for j = 1,#words do input[j]= {t(words[j]), u({0,0})} end
	for j = #words + 1,#tree do input[j]= {t("[-]"), u({0,0})} end
	for j = 1,#words do min[j] = j end
	for j = #words + 1, #tree do min[j] = 1000000 end
	min[0] = 1000000
	input[0]= {"", u({0,0})}
	local diff = 0
	for j = 1,#tree do
		netwrk:forward(input[j])
		diff = correct[j][1][1] -netwrk.output[1][1]
		if math.abs(correct[j][1][1] - netwrk.output[1][1])<0.33 then totalcorrect = totalcorrect + 1 end
		--if bucket(correct[j][1][1]) == bucket(netwrk.output[1][1]) then totalcorrect = totalcorrect + 1 end
		totalerror = totalerror + math.abs(diff);
		total = total + 1;
		--print("\t\t", (correct[j][1][1] - netwrk.output[1][1] ) )
		--input[tree[j]][2] = torch.add( input[tree[j]][2] , netwrk.output[1] )
		if min[j] < min[tree[j]] then
			min[tree[j]] = min[j]
			input[tree[j]][2][1][2] = input[tree[j]][2][1][1]
			input[tree[j]][2][1][1] = newntwk.output[1][1]
		else
			input[tree[j]][2][1][1] = newntwk.output[1][1]
		end
	end
	if bucket(correct[#tree][1][1]) == bucket(netwrk.output[1][1]) then sentcorrect = sentcorrect + 1 end
	senttotal = senttotal + 1
	sentme = sentme + math.abs(diff)
end

function run(words, tree, netwrk)
	local output = {}
	local input = {}
	--local correct = getcorrectvalues(words, tree)
	--local layer = {}
	for j = 1,#words do input[j]= {t(words[j]), u({0,0})} end
	for j = #words + 1,#tree do input[j]= {t("[-]"), u({0,0})} end
	input[0]= {"", u({0,0})}
	--local diff = 0
	out = 0.5
	for j = 1,#tree do
		netwrk:forward(input[j])
		out = netwrk.output[1][1]
	--	diff = correct[j][1][1] -netwrk.output[1][1]
	--	if math.abs(correct[j][1][1] - netwrk.output[1][1])<0.33 then totalcorrect = totalcorrect + 1 end
		--if bucket(correct[j][1][1]) == bucket(netwrk.output[1][1]) then totalcorrect = totalcorrect + 1 end
	--	totalerror = totalerror + math.abs(diff);
	--	total = total + 1;
		--print("\t\t", (correct[j][1][1] - netwrk.output[1][1] ) )
		input[tree[j]][2] = torch.add( input[tree[j]][2] , netwrk.output[1] )
	end
	return out
end

function trainNtwrk(n, name, uc)
	network =  GRU.create(vectorSize, 2)
	for i = 1, n do
		local words = sentences[i]:split("|")
		local tree = trees[i]
		if set[i] == "1" then
			network = train(words, tree, network, uc)
		end
	end
	torch.save("./" .. name, network)
	return network
end

function testNtwrk(n)
	totalcorrect = 0;
	totalerror = 0;
	total = 0;
	sentcorrect = 0;
	senttotal = 0;
	sentme = 0;
	for i = 1, n do
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
end

--load()
--testNtwrk(4000, trainNtwrk(900,"ntwk3", 0.002) )

--function Analyzer.load(str)
--	dload()

--end
--

function sent2(str)
	parser = require 'Parser'
	res = parser.parse(str)
	wor = res:split("@@")[2]:split("|")
	--network1 = trainNtwrk(10,"ntwk3", 0.001)
	network1 = torch.load("./ntwk2")
	tre = map(tonumber, res:split("@@")[1]:split("|") );
	--print(run(wor,tre, network1) );
	return run(wor,tre, network1)
end

function sent(str, name)
	parser = require 'Parser'
	res = parser.parse(str)
	wor = res:split("@@")[2]:split("|")
	--network1 = trainNtwrk(10,"ntwk3", 0.001)
	network1 = torch.load("./" .. name)
	tre = map(tonumber, res:split("@@")[1]:split("|") );
	print(res:split("@@")[1], res:split("@@")[2]);
	return run(wor,tre, network1)
end

--load()
--sent("Ala has a cat.")

return Analyzer
--
