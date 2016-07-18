require 'torch'
require 'nn'
require 'nngraph'
require 'Serv'
require 'Analyzer2'
GRU = require 'GRU2'
Double = require 'Double'
GRU1 = require 'GRU1'

local Analyzer = {}

function load(number)
	SST 		= require 'SST'
	Glove		= require 'Glove'
	set 		= SST.loadSplit()
	results 	= SST.loadResults()
	sentences 	= SST.loadSentences()
	phrases 	= SST.loadPhrases()
	trees 		= SST.loadTrees()
	
	if number == 0 then
		dictionary = Glove.loadDictionary('./glove.6B/glove.6B.50d.txt',true)
	else
		dictionary = Glove.loadDictionary('./glove.6B/glove.6B.50d.txt',true, number)
	end
	vectorSize = #(dictionary["do"]) -- vector size
end

function dload()
	SST 		= require 'SST'
	Glove		= require 'Glove'
	Glove.dset(300)
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
		if priority[j] == nil then priority[j] = j end --print("\t\t",words[j], priority[j])
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


function test(words, tree, netwrk)
	local output = {}
	local input = {}
	local min = {}
	nwords = #words;
	local correct = getcorrectvalues(words, tree)
	local layer = {}
	for j = 1,nwords do input[j]= {t(words[j]), u({0,0})} end
	for j = nwords + 1,#tree do input[j]= {t("[-]"), u({0,0})} end
	for j = 1,nwords do input[j]= {t(words[j]), u({0,0})} end
	for j = nwords + 1,#tree do input[j]= {t("[-]"), u({0,0})} end
	for j = 1,nwords do min[j] = j end
	for j = nwords + 1, #tree do min[j] = 1000000 end
	min[0] = 1000000
	input[0]= {"", u({0,0})}
	local diff = 0
	for j = 1,#tree do
		netwrk:forward(input[j])
		diff = correct[j][1][1] -netwrk.output[1][1]
		--if math.abs(correct[j][1][1] - netwrk.output[1][1])<0.33 then totalcorrect = totalcorrect + 1 end
		if bucket(correct[j][1][1]) == bucket(netwrk.output[1][1]) then totalcorrect = totalcorrect + 1 end
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
	for j = 1,#words do input[j]= {t(words[j]), u({0,0})} end
	for j = #words + 1,#tree do input[j]= {t("[-]"), u({0,0})} end
	input[0]= {"", u({0,0})}
	out = 0.5
	for j = 1,#tree do
		netwrk:forward(input[j])
		out = netwrk.output[1][1]
		input[tree[j]][2] = torch.add( input[tree[j]][2] , netwrk.output[1] )
	end
	return out
end

function sent2(str)
	par = require 'Parser'
	res = par.parse(str)
	print(res)
	wor = res:split("@@")[2]:split("|")
	print(wor)
	--network1 = trainNtwrk(10,"ntwk3", 0.001)
	network1 = torch.load("./networks/ntwk2")
	tre = map(tonumber, res:split("@@")[1]:split("|") );
	--print(run(wor,tre, network1) );
	return run(wor,tre, network1)
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
	torch.save("./networks/" .. name, network)
	return network
end

function testNtwrk(n , net)
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
			test(words,tree, net)
		end
	end
	print(" fraction correct  : \t", totalcorrect/total)
	print(" mean error        : \t", totalerror/total)
	print(" sentences correct : \t", sentcorrect/senttotal)
	print(" sentences mean err: \t", sentme/senttotal)
end

--load()
--testNtwrk(4000, trainNtwrk(900,"ntwk3", 0.002) )

function trainNtwrk2(n, name, uc, hi1, hi2, hi3)
	network = Double.create(vectorSize, 1, hi1, hi2, hi2, hi3, hi3)
	for i = 1, n do
		local words = sentences[i]:split("|")
		local tree = trees[i]
		if set[i] == "1" then
			network = trainII(words, tree, network, network, uc, uc)
		end
	end
	torch.save("./networks/" .. name, network)
	return network
end


function testNtwrk2(n, net)
	totalcorrect = 0; total = 0;
	for i = 1, n do
		local words = sentences[i]:split("|")
		local tree = trees[i]
		if set[i] == "2" then testII(words, tree, net, net) end
	end
	
end


function trainNtwrk3(n, name, uc1, uc2)
	network1 = Double.create(vectorSize, 1, 50, 40, 40, 30, 30)
	network2 = GRU1.create(2, 2)
	for i = 1, n do
		local words = sentences[i]:split("|")
		local tree = trees[i]
		if set[i] == "1" then
			networks = trainII(words, tree, network1, network2, 0.15, uc2)
			network1 = networks[1]
			network2 = networks[2]
		end
	end
	return {network1, network2, network3}
end

function testNtwrk3(network1 , network2)
	statistics = require 'Stat'
	statistics:zero()
	for i = 1, 7000 do
		local words = sentences[i]:split("|"); local tree = trees[i]
		if set[i] == "2" then
			networks = testII(words, tree, network1, network2)
		end
	end
	statistics:printout()
end

function sent(str, name1, name2)
	local par = require 'Parser'
	local res = par.parse(str)
	local wor = res:split("@@")[2]:split("|")
	local tre = map(tonumber, res:split("@@")[1]:split("|") );
	local network1 = torch.load("./networks/" .. name1)
	local network2 = torch.load("./networks/" .. name2)
	--print(res:split("@@")[1], res:split("@@")[2]);
	return execute(wor,tre, network1, network2, true)
end


load(1000000)
--testNtwrk(4000, trainNtwrk(900,"ntwk3", 0.002) ) 
--dload()
--nwk = trainNtwrk2(11000,"ntwkII", 0.15, 50, 40, 30); testNtwrk2(11000, nwk)
--print(" fraction correct  : \t", totalcorrect/total)

--nwks = trainNtwrk3(2000, "ntwkIIbis", 0.15, 0.04)
--testNtwrk3(nwks[1], nwks[2])
--torch.save("networks/main2",nwks[2])

--treetrainer(2000, "main2b", 0.04, true, true)
--wordtrainer("main1b",0.15, true)


sent("Ala has a cat.", "main1", "main2")
sent("Ala has a stupid cat.", "main1", "main2")
sent("Ala has a very stupid cat.", "main1", "main2")
sent("Ala has a completely stupid cat.", "main1", "main2")

sent("It is difficult for the isolated individual to work himself out of the immaturity which has become almost natural for him.", "main1", "main2")
sent("This i so boring.", "main1", "main2")


return Analyzer
