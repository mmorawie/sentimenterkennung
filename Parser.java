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

public class Parser {
    private final static String PCG_MODEL = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
    private final TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(), "invertible=true");
    private final LexicalizedParser parser = LexicalizedParser.loadModel(PCG_MODEL);
	private static int index = 0;

    public Tree parse(String str) {
        List<CoreLabel> tokens = tokenize(str);
        Tree tree = parser.apply(tokens);
        return tree;
    }

    private List<CoreLabel> tokenize(String str) {
        Tokenizer<CoreLabel> tokenizer = tokenizerFactory.getTokenizer(new StringReader(str));
		//Tokenizer<CoreLabel> tokenizer = PTBTokenizer.factory(new CoreLabelTokenFactory(), "invertible=true").getTokenizer(new StringReader(str));
        return tokenizer.tokenize();
		//return null;
    }
	
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

	public static String process(String str){
		Parser parser = new Parser();
        Tree tree = parser.parse(str);
        List<Tree> leaves = tree.getLeaves();
		TreeBinarizer bin = TreeBinarizer.simpleTreeBinarizer(
			new SemanticHeadFinder(),
			new PennTreebankLanguagePack());
		tree = bin.transformTree(tree);
		leaves = tree.getLeaves();
		//System.out.println("--------------------" + tree.getChild(0));
		Node root = DFScopy(tree);
		//DFSprint(root, 1);
		index = 0; 
		DFSnumber(root); 
		DFSpostorder(root);
		//DFSprint(root, 1);
		int[] arr = new int[index]; 
		DFSparent(root, arr);

		String ppr = "" + arr[1];
		for (int i = 2; i < index-1; i++) ppr = ppr + "|" + arr[i];
		ppr = ppr + "|0";

		String sentence = leaves.get(0).label().value();
		for (int i = 1; i<leaves.size(); i++) sentence = sentence + "|" + leaves.get(i).label().value();

		/*for (Tree leaf : leaves) leaf.parent(tree).setChildren(new Tree[0]);
		leaves = tree.getLeaves();
		int[] parent = new int[tree.size()];
		int size = tree.size();
		Tree root = leaves.get(0);

		while(true){
			if( root.label().value().equals("ROOT") ) break;
			root = root.parent(tree);
		}
		dfs(root, size-1, parent);
		leaves = tree.getLeaves();
		int j = 0;
		for (Tree leaf : leaves) {
			parent[j] = Integer.parseInt(leaf.parent(tree).label().value());
			j = j+1;
		}
		String ppr = parent[0] + "";
		for (int i = 1; i<parent.length; i++) ppr = ppr + "|" + parent[i];*/

		
        return (ppr + "@@"  + sentence);
	}

	public static Node DFScopy(Tree t){
		Tree[] arr = t.children();
		Node node = new Node();
		node.text = t.label().value();
		for (Tree child : arr) {
			Tree child2 = child;
			while (child2.children().length == 1) child2 = child2.children()[0];
			node.child.add( DFScopy(child2) );
		}
		return node;
	}

	public static void DFSprint(Node node, int z){
		for (int i = 0; i < z; i++) System.out.print("|");
		System.out.println(node.text + " " + node.nbr);
		for (Node child : node.child) {
			DFSprint(child, z+1);
		}
		return;
	}

	public static void DFSnumber(Node node){
		for (Node child : node.child) {
			DFSnumber(child);
		}
		if (node.child.size() == 0){
			index++;
			node.nbr = index;
		}
		return;
	}

	public static void DFSpostorder(Node node){
		for (Node child : node.child) {
			DFSpostorder(child);
		}
		if (node.nbr == 0){
			index++;
			node.nbr = index;
		}
		return;
	}

	public static void DFSparent(Node node, int[] array){
		for (Node child : node.child) {
			child.parent = node.nbr;
			array[child.nbr] = node.nbr;
			DFSparent(child, array);
		}
		return;
	}

	public static void hello(int a){
		System.out.println(" hello : " + a);
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
		t.label().setValue(""+n);
	}

	static class Node{
		public String text;
		public int nbr;
		public int parent;
		ArrayList<Node> child = new ArrayList<Node>();

	}


}
