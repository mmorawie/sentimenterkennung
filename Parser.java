import java.io.StringReader;
import java.util.List;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.parser.lexparser.TreeBinarizer;
import java.util.*;
import edu.stanford.nlp.ling.*;

class Parser {
    private final static String PCG_MODEL = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
    private final TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(), "invertible=true");
    private final LexicalizedParser parser = LexicalizedParser.loadModel(PCG_MODEL);

    public Tree parse(String str) {
        List<CoreLabel> tokens = tokenize(str);
        Tree tree = parser.apply(tokens);
        return tree;
    }

    private List<CoreLabel> tokenize(String str) {
        Tokenizer<CoreLabel> tokenizer =
            tokenizerFactory.getTokenizer(
                new StringReader(str));
        return tokenizer.tokenize();
    }

    //public static void exe(String str) {
	public static void main(String argv[]) {
        String str = argv[0];
        Parser parser = new Parser();
        Tree tree = parser.parse(str);

        List<Tree> leaves = tree.getLeaves();

		TreeBinarizer bin = TreeBinarizer.simpleTreeBinarizer(
			new SemanticHeadFinder(),
			new PennTreebankLanguagePack());
		//System.out.println(tree);
		tree = bin.transformTree(tree);

		leaves = tree.getLeaves();
		String sentence = leaves.get(0).label().value();
		for (int i = 1; i<leaves.size(); i++)
			sentence = sentence + "|" + leaves.get(i).label().value();

		for (Tree leaf : leaves) {
            //System.out.print(leaf.label().value() + "|");
			leaf.parent(tree).setChildren(new Tree[0]);
		}
		//System.out.println(tree);
		leaves = tree.getLeaves();
		int[] parent = new int[tree.size()];
		int size = tree.size();
		Tree root = leaves.get(0);

		while(true){
			if( root.label().value().equals("ROOT") ) {
				break;
			}
			root = root.parent(tree);
		}
		dfs(root, size-1, parent);
		//System.out.println(tree);
		leaves = tree.getLeaves();
		int j = 0;
		for (Tree leaf : leaves) {
			//System.out.println (leaf.label().value() + " --- " + leaf.parent(tree).label().value());
			parent[j] = Integer.parseInt(leaf.parent(tree).label().value());
			j = j+1;
		}
		String ppr = parent[0] + "";
		for (int i = 1; i<parent.length; i++){
			ppr = ppr + "|" + parent[i];
		}

        System.out.println("\n"+ ppr + "@@"  + sentence);
    }

	public static boolean has(ArrayList<Tree> list, Tree t){
		for (Tree tt : list) {
            if(tt == t) return true;
        }
		return false;
	}

	public static void dfs(Tree t, int n, int[] parent){
		Tree[] arr = t.children();
		int i = n;
		for (Tree child : arr) {
			//if(child.children().length == 1) child.setChildren(child.children()[0].children());
			if(child.children().length>0){
				i = i-1;
				dfs(child, i, parent);
				parent[i] = n;
			}
		}
		//System.out.println(t.label().value() + "<---" + n);
		t.label().setValue(""+n);
	}

	static class Node{
		public String text;
		public int nbr;
		public int parent;
		ArrayList<Node> child = new ArrayList<Node>();
		Tree tree;
	}
}
