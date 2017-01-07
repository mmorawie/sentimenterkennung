

import java.awt.EventQueue;
import javax.swing.JFrame;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.*;


class Node extends JPanel{
    public JLabel main;
    public static Color[] colorset = {
        Color.decode("#d7191c"),
        Color.decode("#fdae61"),
        Color.decode("#ffffbf"),
        Color.decode("#abd9e9"),
        Color.decode("#2c7bb6")
    };
    
    public Node(String text, int sent){
        main = new JLabel(text, SwingConstants.CENTER){
            @Override
            public Dimension getMaximumSize() {
                Dimension d = super.getMaximumSize();
                d.width = Integer.MAX_VALUE;
                d.height = Integer.MAX_VALUE;
                return d;
            }
        };

        main.setFont(new Font("Monospace", Font.PLAIN, 20)); 
        main.setForeground(Color.black);
        main.setToolTipText(sent + "");
        main.setBorder(new CompoundBorder( main.getBorder() , BorderFactory.createMatteBorder(6, 6, 6, 6, Color.BLACK) ));
        main.setOpaque(true);
        main.setBackground( colorset[sent] );
        this.setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        this.add(main); main.setAlignmentX(Component.CENTER_ALIGNMENT); 
        this.setBackground(Color.BLACK);
    }

    public Node(String text, int sent, Node left, Node right){
        this(text, sent);
        JPanel zebra = new JPanel();
        zebra.setBackground(Color.BLACK);
        zebra.setLayout(new BoxLayout(zebra, BoxLayout.X_AXIS));
        this.add(zebra);
        zebra.add( left );
        zebra.add( right );
    }
}

class Leaf extends JLabel{
    public Leaf(String text, int sent){  
        this.setText(text);
        this.setFont(new Font("Monospace", Font.PLAIN, 20)); 
        this.setBorder(new CompoundBorder( this.getBorder() , new EmptyBorder(10,10,10,10) ));
        this.setOpaque(true);
        this.setBackground(new Color(128,128,0));
    } 
}

class Graph {
    static JFrame frame;
    static JLabel line; 
    public static Node[] nodes;
    

    public static void main(String[] args){
        frame = new JFrame("aaa");
        frame.setSize(new Dimension(800,600));
        //JLabel mainLabel = new JLabel("aaa\n bbb");
        
        //frame.setLayout(new BoxLayout(frame, BoxLayout.PAGE_AXIS));
        //frame.add(Box.createVerticalGlue());
        //frame.add(mainLabel);
        
        //nextLine();
        //nextPhrase("aaaaa");
        //
        frame.setLayout(new BorderLayout());

        init(5);
        setLeaf("Not", 1, 0);
        setLeaf("in ", 1, 1);
        setLeaf("here", 1, 2);

        setNode("in here", 1, 3, 1, 2);
        setNode("Not in here", 1, 4, 0, 3);

        frame.add(nodes[4], BorderLayout.CENTER );
        frame.setVisible(true);


        /*Container c = frame.getContentPane();
        BufferedImage im = new BufferedImage(c.getWidth(), c.getHeight(), BufferedImage.TYPE_INT_ARGB);
        c.paint(im.getGraphics());
        try {
            ImageIO.write(im, "PNG", new File("shot.png"));
        } (catch Exceprion ioe) {

        }*/
        return;
    }

    public static void init(int n){
        nodes = new Node[n]; 
    }

    public static void setLeaf(String text, int sent, int i){
        nodes[i] = new Node(text, sent);
    }

    public static void setNode(String text, int sent, int i, int j, int k){
        nodes[i] = new Node(text, sent, nodes[j], nodes[k]);
    }

    public static void display(int i){
        frame = new JFrame("Parse tree");
        frame.setSize(new Dimension(800,600));
        JPanel fill = new JPanel();
        fill. setBackground(Color.BLACK);
        fill.setOpaque(true);
        //frame.setBackground(Color.BLACK);
        frame.setLocationRelativeTo(null);
        //frame.getContentPane().setBackground(Color.BLACK);

        frame.setLayout(new BorderLayout());
        fill.setLayout(new BorderLayout());
        frame.add( fill, BorderLayout.CENTER );
        fill.add(nodes[nodes.length-1], BorderLayout.CENTER );
        frame.setVisible(true);
        try {
            //Thread.sleep(10);
        } catch (Exception fuckingException) {

        }
        //System.out.println("ok -- -- ");
    }
}
