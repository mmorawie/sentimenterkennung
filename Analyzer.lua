require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'xlua' 
require 'sys' 
require 'lfs' 
require 'penlight'
Double = require 'Double'
Node = require 'Node'


local Analyzer = {}

function loadsets()
	set 		= SST.loadSplit()
	results 	= SST.loadResults()
	sentences 	= SST.loadSentences()
	phrases 	= SST.loadPhrases()
	trees 		= SST.loadTrees()
	testset = {}; a = 1;
	trainset = {}; b = 1;
	devset = {}; c = 1;
	for i = 1, #sentences do
		if set[i] == "1" then trainset[a] = i; a = a + 1 end
		if set[i] == "2" then testset[b] = i; b = b + 1 end
		if set[i] == "3" then devset[c] = i; c = c + 1 end
	end
end

function load(number)
	SST 		= require 'SST'
	Glove		= require 'Glove'
	if number == 0 then
		dictionary = Glove.loadDictionary('./glove.6B/glove.6B.50d.txt',true)
	else
		dictionary = Glove.loadDictionary('./glove.6B/glove.6B.50d.txt',true, number)
	end
	loadsets()
	vectorSize = #(dictionary["do"]) -- vector size
end

function zload()
	SST 		= require 'SST'
	Glove		= require 'Glove'
	loadsets()
	dictionary = {}
	for i = 1, #sentences do
		local wrd = sentences[i]:split('|')
		for j = 1, #wrd do dictionary[ wrd[j] ] = -1 end
	end
	dictionary = Glove.lookup('./glove.6B/glove.6B.50d.txt')
	vectorSize = #(dictionary["do"]) -- vector size
end

function dload()
	SST 		= require 'SST'
	Glove		= require 'Glove'
	--Glove.dset(300)
	vectorSize = 300
	dictionary = {}
	loadsets()
	vectorSize = 300
end

function sent(str, name)
	local par = require 'Parser'
	local res = par.parse(str)
	local wor = res:split("@@")[2]:split("|")
	local tre = map(tonumber, res:split("@@")[1]:split("|") );
	local no = Node.setup(wor, tre)
	local network = torch.load("./networks/" .. name)
	network:forward(no[#no])
	no[#no]:printout()
end

function score(set, mod)
	local no
	local sc = 0
	local ok = 0
	local loss = 0
	for i = 1, #set do
		mod.bTree:evaluate()
		no = Node.setup(sentences[set[i]]:split('|'), trees[set[i]])
		_, loss = mod.bTree:forward(no[#no])
		--print(loss)
		if ( tablebucket( no[#no].output ) ==  bucket( no[#no].correct ) ) then ok = ok + 1 end
		sc = sc + loss
		mod.bTree:clean(no[#no])
	end
	return ok, sc
end

include('Trainer.lua')
include('bTree.lua')
include('Model.lua')
GRU4 = require 'GRU'
statistics = require 'Stat'

classnumber = 5
require 'Service'



--dload()
zload(1000000)


--local nodee = Node.setup(sentences[testset[2]]:split('|'), trees[testset[2]])
Parser = require 'Parser'
--Parser.drawTree(nodee)



local model = Trainer({}, 0.05)
local best = -1.0
local finalmodel = model
for i = 1, 5 do --epochs
	model.bTree:training()
	model:train(#trainset)
	local ok1, score1 = score(trainset, model)
  	local ok2, score2 = score(devset, model)
  	print('{', (1.0*score2)/#devset , ',', (1.0*score1)/#trainset, ',', (1.0*ok2)/#devset ,',',  (1.0*ok1)/#trainset , '},')
  	if ok2 > best then
		best = ok2
		finalmodel = Trainer{}
		finalmodel.params:copy(model.params)
	end 
end
print('finished training')



statistics:zero()
for i = 1, #testset do
	model.bTree:evaluate()
	local no = Node.setup(sentences[testset[i]]:split('|'), trees[testset[i]])
	finalmodel.bTree:forward(no[#no])
	--print( no[#no].output, tablebucket(no[#no].output),  no[#no].correct )
		statistics:sentences(no[#no].output, no[#no].correct)
		for i = 1, no.wsize do statistics:words(no[i].output, no[i].correct) end
		for i = no.wsize + 1, no.tsize - 1 do statistics:phrases(no[i].output, no[i].correct, no[i].size) end
	finalmodel.bTree:clean(no[#no])
end
statistics:printout()

--torch.save('./networks/' .. 'module4', model.bTree) 


--wordtrainer("main1b",0.05, true)
--require 'Analyzer4'
--trainer()

--sent('Ala has a cat.', 'module1')
--treetrainer(2000, "main2b", 0.04, true, true)

--wordtrainer("main1b",0.85, true)


--sent("Ala has a cat.", "main1", "main2")
--sent("Ala has a stupid cat.", "main1", "main2")
--sent("Ala has a very stupid cat.", "main1", "main2")
--sent("Ala has a completely stupid cat.", "main1", "main2")
--sent("It is difficult for the isolated individual to work himself out of the immaturity which has become almost natural for him.", "main1", "main2")
--sent("This is so boring.", "main1", "main2")

return Analyzer 