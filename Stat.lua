

Stat = {
    box = 2,
    maxlen = 80,
    limit = 80/2,
    zeroc = function(self)
        self.cnt = 0
        self.tot = 0
    end,
    zero = function(self)
        self.wordscorrect = 0
        self.phrasescorrect = 0
        self.sentencescorrect = 0
        self.totalcorrect = 0

        self.wordscorrect2 = 0
        self.phrasescorrect2 = 0
        self.sentencescorrect2 = 0
        self.tw2 = 0
        self.tp2 = 0
        self.ts2 = 0

        self.nwords = 0
        self.nphrases = 0
        self.nsentences = 0
        self.ntotal = 0

        self.wordserror = 0
        self.phraseserror = 0
        self.sentenceserror = 0
        self.totalerror = 0

        self.lenghts = {}
        self.lenghtscorrect = {}
        for i = 1, 100 do 
            self.lenghts[i] = 0 
            self.lenghtscorrect[i] = 0
        end
    end,
    conde = function(rea)
            return bucket(rea) > 2
        end,
    words = function(self, out, rea)
        if tablebucket(out) == bucket(rea) then 
            self.wordscorrect = self.wordscorrect + 1
            self.totalcorrect = self.totalcorrect + 1
            if self.conde(rea) then self.wordscorrect2 = self.wordscorrect2 + 1  end
        end
        self.nwords = self.nwords + 1
        if self.conde(rea) then self.tw2 = self.tw2 + 1 end
    end,
    sentences = function(self, out, rea)
        if tablebucket(out) == bucket(rea) then 
            self.sentencescorrect = self.sentencescorrect + 1
            self.phrasescorrect = self.phrasescorrect + 1
            if self.conde(rea) then self.sentencescorrect2 = self.sentencescorrect2 + 1  end
            if self.conde(rea) then self.phrasescorrect2 = self.phrasescorrect2 + 1  end
        end
        self.nsentences = self.nsentences + 1
        self.nphrases = self.nphrases + 1
        if self.conde(rea) then self.ts2 = self.ts2 + 1 end
        if self.conde(rea) then self.tp2 = self.tp2 + 1 end
    end,
    phrases = function(self, out, rea, size)
        if tablebucket(out) == bucket(rea) then 
            self.phrasescorrect = self.phrasescorrect + 1 
            if self.conde(rea) then self.phrasescorrect2 = self.phrasescorrect2 + 1  end
        end
        self.nphrases = self.nphrases + 1
        if self.conde(rea) then self.tp2 = self.tp2 + 1  end
        for i = 1,self.limit do
            if (i-1)*self.box < size and i*self.box >= size then
                self.lenghts[i] = self.lenghts[i] + 1
                if tablebucket(out) == bucket(rea) then self.lenghtscorrect[i] = self.lenghtscorrect[i] + 1 end
                break
            end
        end
    end,
    count = function(self, a)
        self.cnt = self.cnt + a;
        self.tot = self.tot + 1;
    end,
    printout = function(self)
        print("\n sentences = ", self.sentencescorrect/self.nsentences)
        print("words =     ", self.wordscorrect/self.nwords)
        print("phrases =   ", self.phrasescorrect/self.nphrases)
        print("\n sentences 2 = ", self.sentencescorrect2/self.ts2)
        print("words =    2  ", self.wordscorrect2/self.tw2)
        print("phrases =  2  ", self.phrasescorrect2/self.tp2)
        io.write("{")
        for i = 1, self.limit do
            if self.lenghts[i] == 0 then break end
            io.write("{" .. i*self.box .. ", " .. self.lenghtscorrect[i]/self.lenghts[i] .. "},\n")
        end
        io.write("};\n")
    end,
    cntstr = function(self)
       return "count = " .. tostring(self.cnt/self.tot)
    end
}





return Stat

