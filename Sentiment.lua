

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'xlua' 
require 'sys' 
require 'lfs' 
require 'penlight'

include('Trainer.lua')
include('Model.lua')
GRU4 = require 'GRU'
Single = require 'Single'
Output = require 'Output'
statistics = require 'Stat'
classnumber = 5
Service = require 'Service'
Node = require 'Node'


local Sentiment = {
    verbose = true,
	classnumber = 5,
	vector_size = nil,
	dictionary = {},
	dict_max_size = nil,
	dict_size = 0
}

Service.parent = Sentiment

function Sentiment.set_verbose(true_or_false)
	if true_or_false ~= true and true_or_false ~= false then error('expected true or false') end
	if true_or_false then
		Sentiment.verbose = true
	else 
		Sentiment.verbose = false
	end
end

function Sentiment.load_vocabulary() 
	local dict = {}
	local numberofwords = 0
	Sentiment.SST.vocabulary = {}
	Sentiment.SST.ivocabulary = {}
	for i = 1, #(Sentiment.SST.sentences) do
		local wrd = Sentiment.SST.sentences[i]:split('|')
		for j = 1, #wrd do dict[ wrd[j] ] = -1 end
	end
    for a, b in pairs(dict) do 
		numberofwords = numberofwords + 1 
		Sentiment.SST.vocabulary[numberofwords] = a
		Sentiment.SST.ivocabulary[a] = numberofwords
	end
	Sentiment.glove_reprezentation = torch.Tensor(numberofwords, Sentiment.vector_size)
	for i = 1, #Sentiment.SST.vocabulary do
		Sentiment.glove_reprezentation[i] = t( Sentiment.SST.vocabulary[i] )
	end	
end

function Sentiment.load_binset()
	Sentiment.binSST = {}
	Sentiment.binSST.set = Sentiment.SST.set 
	Sentiment.binSST.results 	= Sentiment.SST.results 
	Sentiment.binSST.sentences = Sentiment.SST.sentences
	Sentiment.binSST.phrases 	= Sentiment.SST.phrases
	Sentiment.binSST.trees = Sentiment.SST.trees 
	Sentiment.binSST.ivocabulary = Sentiment.SST.ivocabulary
	Sentiment.binSST.vocabulary = Sentiment.SST.vocabulary
	Sentiment.binSST.trainset = {}; local a = 1;
	Sentiment.binSST.testset = {}; local b = 1;
	Sentiment.binSST.devset = {}; local c = 1;

	for i = 1, #Sentiment.SST.sentences do
		no = Node.setup(Sentiment.SST.sentences[i]:split('|'), Sentiment.SST.trees[i], Sentiment.SST)
		if  no[#no].correct < 0.4 or  no[#no].correct > 0.6 then
			if Sentiment.SST.set[i] == "1" then Sentiment.binSST.trainset[a] = i; a = a + 1 end
			if Sentiment.SST.set[i] == "2" then Sentiment.binSST.testset[b] = i; b = b + 1 end
			if Sentiment.SST.set[i] == "3" then Sentiment.binSST.devset[c] = i; c = c + 1 end
		end
	end
end 

function Sentiment.fload(dim, number)
	Sentiment.SST 		= require 'SST'; 
	local Glove			= require 'Glove'
	if dim ~= nil and dim ~= 50 and dim ~= 100 and dim ~= 200 and dim ~= 300 then error('wrong dim value') end
	if number < 0 then error('wrong number value') end
	Sentiment.vector_size = dim 
	if dim == nil then Sentiment.vector_size = 300 end
	Sentiment.SST.loadsets()
	if number == 0 then Sentiment.load_vocabulary() ; Sentiment.load_binset();  return end
	if dim == nil then
		Sentiment.dictionary, dict_size = Glove.loadDictionary(Glove.dir1 ..'/glove.6B.300d.txt', Sentiment.verbose, number)
	else
		Sentiment.dictionary, dict_size = Glove.loadDictionary(Glove.dir2 ..'/glove.6B.' .. dim .. 'd.txt', Sentiment.verbose, number)
	end
	Sentiment.load_vocabulary() 
	Sentiment.load_binset()
end

function Sentiment.zload(dim)
	Sentiment.SST 		= require 'SST'
	local Glove			= require 'Glove'
	if dim ~= nil and dim ~= 50 and dim ~= 100 and dim ~= 200 and dim ~= 300 then error('wrong dim value') end
	Sentiment.vector_size = dim 
	if dim == nil then Sentiment.vector_size = 300 end
	Sentiment.SST.loadsets()
	for i = 1, #Sentiment.SST.sentences do
		local wrd = Sentiment.SST.sentences[i]:split('|')
		for j = 1, #wrd do Sentiment.dictionary[ wrd[j] ] = -1 end
	end
	if dim == nil then
		Sentiment.dictionary, dict_size = Glove.lookup(Glove.dir1 ..'/glove.6B.300d.txt', Sentiment.dictionary)
	else
		Sentiment.dictionary, dict_size = Glove.lookup(Glove.dir2 ..'/glove.6B.' .. dim .. 'd.txt', Sentiment.dictionary)
	end
	Sentiment.load_vocabulary() 
	Sentiment.load_binset()
end

local function score(set, mod, bucket)
	local no
	local sc = 0
	local ok = 0
	local loss = 0
	for i = 1, #set do
		mod.Model:evaluate()
		no = Node.setup(Sentiment.SST.sentences[set[i]]:split('|'), Sentiment.SST.trees[set[i]], Sentiment.SST)
		_, loss = mod.Model:forward(no, Sentiment.SST)
		if ( tablebucket( no[#no].output ) ==  bucket( no[#no].correct ) ) then ok = ok + 1 end
		sc = sc + loss
		mod.Model:clean(no[#no])
	end
	return ok, sc
end


function Sentiment.fg_train(epochs, mem, learning_rate, regularization, options)
	local options = options or {}
	options.vector_size = Sentiment.vector_size
	options.glove_T = Sentiment.glove_reprezentation
	options.sst = Sentiment.SST
	options.bucket = function (x)
        if x<0.2 then return 0 end
        if x<0.4 then return 1 end
        if x<0.6 then return 2 end
        if x<0.8 then return 3 end
	    return 4;
    end
	return Sentiment.train(5, epochs, mem, learning_rate, regularization, options)
end

function Sentiment.bin_train(epochs, mem, learning_rate, regularization, options)
	local options = options or {}
	options.vector_size = Sentiment.vector_size
	options.glove_T = Sentiment.glove_reprezentation
	options.sst = Sentiment.binSST
	options.bucket = function (x)
        if x<0.4 then return 1 end
        if x<0.6 then return 0 end
	    return 2;
    end
	return Sentiment.train(3, epochs, mem, learning_rate, regularization, options)
end


function Sentiment.train(problem, epochs, mem, learning_rate, regularization, options)
	local model = Trainer(mem,  problem, learning_rate, regularization or 0.0001, options)
    
	local best = -1.0
    local finalmodel = model
	local history = {}
	local trainset = options.sst.trainset
	local devset = options.sst.devset
	for i = 1, epochs do
        model.Model:training()
        model:train(options.sst)
        local ok1, score1 = score(options.sst.trainset, model, options.bucket)
        local ok2, score2 = score(options.sst.devset, model, options.bucket)
		history[i] = {
			dev_loss = (1.0*score2)/#options.sst.devset, 
			train_loss = (1.0*score1)/#options.sst.trainset, 
			dev_ok = (1.0*ok2)/#options.sst.devset,  
			train_ok = (1.0*ok1)/#options.sst.trainset
		}
        if Sentiment.verbose then 
			io.write(string.format("epoch %d : loss = %.4f ,  %.4f ; score =  %.4f ,  %.4f \n", 
				i,
				(1.0*score2)/#options.sst.devset,
				(1.0*score1)/#options.sst.trainset,
				(1.0*ok2)/#options.sst.devset,
				(1.0*ok1)/#options.sst.trainset
			))
		end
		if ok2 > best then
            best = ok2
            finalmodel = Trainer(mem,  problem, learning_rate, regularization or 0.0001, options)
            finalmodel.params:copy(model.params)
        end 
    end
    if Sentiment.verbose then print('finished training') end
	model.Model.number_of_classes = 5
    return history, model.Model
end

function Sentiment.test(model, set)
	if set == nil then set = Sentiment.SST.testset end
	statistics:zero()
	statistics.bucket = model.bucket
	for i = 1, #set do
		model:evaluate()
		local no = Node.setup(Sentiment.SST.sentences[set[i]]:split('|'), Sentiment.SST.trees[set[i]], Sentiment.SST)
		if model.binary ~= true or model.bucket( no[#no].correct ) ~= 1  then 
			model:forward(no, Sentiment.SST)
				statistics:sentences(no[#no].output, no[#no].correct)
				for i = 1, no.wsize do statistics:words(no[i].output, no[i].correct) end
				for i = no.wsize + 1, no.tsize - 1 do statistics:phrases(no[i].output, no[i].correct, no[i].size) end
			model:clean(no[#no])
		end
	end
	return statistics:over() --statistics:printout()
end

function Sentiment.sent(str, model)
	local par = require 'Parser'
	local res = par.parse(str)
	local wor = res:split("@@")[2]:split("|")
	local tre = map(tonumber, res:split("@@")[1]:split("|") );
	local no = Node.setup_2(wor, tre)
	model:forward(no, Sentiment.SST)
	if Sentiment.verbose then no[#no]:printout() end
	for i = 1, #no do no[i].sentiment = tablebucket(no[i].output) end
	return no[#no]
end

function Sentiment.


g_sent(str, model)
	local par = require 'Parser'
	local res = par.parse(str)
	local wor = res:split("@@")[2]:split("|")
	local tre = map(tonumber, res:split("@@")[1]:split("|") );
	local no = Node.setup_2(wor, tre)
	model:forward(no, Sentiment.SST)
	par.drawTree(no)
	for i = 1, #no do no[i].sentiment = tablebucket(no[i].output) end
	return no[#no]
end

Sentiment.fload(50,0)

hist, modell = Sentiment.fg_train(1, 5, 0.05, 0.0001)

--hist, modell2 = Sentiment.bin_train(1, 5, 0.05, 0.0001)

--modell = torch.load('modell')
--modell2 = torch.load('modell2')
--Sentiment.test(modell)
--Sentiment.test(modell2)

--torch.save('modell', modell)
--torch.save('modell2', modell2)

--[[n = Sentiment.sent('This is splendid.' , modell)
n:printout()
n2 = Sentiment.sent('This is splendid.' , modell2)
Sentiment.g_sent('This is splendid.' , modell2)
n2:printout()]]


return Sentiment




