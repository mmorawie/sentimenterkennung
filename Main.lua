

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'xlua' 
require 'sys' 
require 'lfs' 
require 'penlight'
include('Trainer.lua')
include('bTree.lua')
include('Model.lua')
GRU4 = require 'GRU'
statistics = require 'Stat'
classnumber = 5
require 'Service'
Double = require 'Double'
Node = require 'Node'


Sentiment = {
    verbose = true

}

function loadProp()
	local file = io.open('./Directories.prop','r'); io.input(file)
	Sentiment.GloveDirectory = io.read():split('=')[2]
    Sentiment.SSTDirectory = io.read():split('=')[2]
end

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

function load(dim, number)
	SST 		= require 'SST'
	Glove		= require 'Glove'
	loadProp()
	if number == nil then
		dictionary = Glove.loadDictionary(GloveDirectory ..'/glove.6B.' .. dim .. 'd.txt', Sentiment.verbose)
	else
		dictionary = Glove.loadDictionary(GloveDirectory ..'/glove.6B.' .. dim .. 'd.txt', Sentiment.verbose)
	end
	loadsets()
	vectorSize = dim -- #(dictionary["do"]) -- vector size
end

function zload(dim)
	SST 		= require 'SST'
	Glove		= require 'Glove'
	loadsets()
	dictionary = {}
	loadProp()
	for i = 1, #sentences do
		local wrd = sentences[i]:split('|')
		for j = 1, #wrd do dictionary[ wrd[j] ] = -1 end
	end
	print( Sentiment.GloveDirectory .. '/glove.6B.' .. dim ..  'd.txt') 
	dictionary = Glove.lookup( Sentiment.GloveDirectory .. '/glove.6B.' .. dim ..  'd.txt')
	vectorSize = #(dictionary["do"]) -- vector size
end

function dload()
	SST 		= require 'SST'
	Glove		= require 'Glove'
	loadProp()
	vectorSize = 6
	dictionary = {}
	loadsets()
	if Sentiment.verbose then print('< debug load >') end
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



function train(epochs, memory, learning_rate, regularization)
    local model = Trainer(memory, learning_rate)
    local best = -1.0
    local finalmodel = model
    for i = 1, epochs do --epochs
        model.bTree:training()
        model:train(#trainset)
        local ok1, score1 = score(trainset, model)
        local ok2, score2 = score(devset, model)
        if Sentiment.verbose then print('epoch ', i, ' : ', (1.0*score2)/#devset , ',', (1.0*score1)/#trainset, ',', (1.0*ok2)/#devset ,',',  (1.0*ok1)/#trainset ) end
        if ok2 > best then
            best = ok2
            finalmodel = Trainer(memory, learning_rate)
            finalmodel.params:copy(model.params)
        end 
    end
    if Sentiment.verbose then print('finished training') end
    return model
end

function test(model)
	statistics:zero()
	for i = 1, #testset do
		model.bTree:evaluate()
		local no = Node.setup(sentences[testset[i]]:split('|'), trees[testset[i]])
		finalmodel.bTree:forward(no[#no])
			print( no[#no].output, tablebucket(no[#no].output),  no[#no].correct )
			statistics:sentences(no[#no].output, no[#no].correct)
			for i = 1, no.wsize do statistics:words(no[i].output, no[i].correct) end
			for i = no.wsize + 1, no.tsize - 1 do statistics:phrases(no[i].output, no[i].correct, no[i].size) end
			finalmodel.bTree:clean(no[#no])
	end
	statistics:printout()
end



dload(50)
train(5, 10, 0.05, 0.0001)

