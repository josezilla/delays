# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) DS Styleguide

While this isn't an all-inclusive styleguide, we've outlined some general rules to follow when creating materials, in order to be consistent.

If there are any really major issues or significantly impactful reasons to change, add, or remove rules from our shared styleguide, make a Github issue & shop it around – let's get feedback from instructors & see what others think.

<!-- MarkdownTOC -->

- [Important Terms](#important-terms)
- [Default Style](#default-style)
- [Default names for files & folders](#default-names-for-files--folders)
- [Default Screenshots](#default-screenshots)
- [File Structure](#file-structure)
- [Material Metadata](#material-metadata)
- [Headings & Text](#headings--text)
- [Contributing Materials](#contributing-materials)

<!-- /MarkdownTOC -->

<a name="important-terms"></a>
## Important Terms

##### Opening

This is what we call the beginning of a lesson, where we review prior lessons/prework/homework, discuss current objectives, and answer outstanding questions.

##### Introduction

This is what we'll call the beginning of a lesson, when we're mostly just lecturing. It'll probably include the history of the topic, how it connects to other topics we've covered, and why it's important.

##### Demo

This is when we'll be showing something while we do it. We can also throw in interactive techniques or ask students to perform a task along with us.

##### Guided Practice

This is when we're asking students to do something specific using the information we've just given. Walk students through solving a problem or applying this topic to a real world scenario. Solving or understanding this should require the use of the lesson topic (in addition to any prior topics).

##### Independent Practice

This is a name for the chunks of a lesson when you're asking students to create a deliverable on their own. It is often used closer to the end of a lesson (though not necessarily), requiring the use of skills learned in the demo and guided practice sessions. It is intended to encapsulate the information and can act as a miniature or practice version of the unit project.

##### Conclusion

This is where we review the deliverable(s) from the independent practice, recap the lesson objectives, answer any outstanding questions, and discuss homework/upcoming projects/etc.

---

<a name="default-style"></a>
## Default Style

### Voice
- **Projects**: Project materials should be written directly for students.
- **Lessons**: While lesson materials are technically instructor facing, they should primarily be written in a student-facing voice for consistency. If you want to make specific suggestions for instructors, use the "instructor's note" format:

> Instructor Note: Here is a sample note that is only intended for instructors.


### Use dashes, not underscores
While dashes & underscores are both fine choices, our structure starts looking ugly when we randomly have both, depending on preference. Underscores are wonderful, no hating on underscores.

But dashes are more commonly associated with web URLs, and in some environments allow us to jump using the keyboard, where underscores don't always let us do that.

Discounting any automatically generated code where underscores are absolutely necessary, **use dashes for folder & filenames.**

### Use leading zeros by default

When you have multiple files sitting next to each other in a folder, number them with one or more leading zeros for visual consistency.

```
- 01-what-is-data-science
- 02-research-design-and-pandas
- 03-statistics-fundamentals
```

### Lowercase file names (and use dashes)

By consistently keeping everything lowercase, we won't have to hit shift when navigating in terminal. It'll save us a tiny fraction of time, but the consistency will save us a lot of guessing.

---

<a name="default-names-for-files--folders"></a>
## Default names for files & folders

Let's use file & folder naming defaults that give students good practice.

> This needs to be filled in

<a name="default-screenshots"></a>
## Default Screenshots

### Full Window Screenshots
Use a quick `cmd+shift+4` and hit `space` to **grab a Full Window screenshot** (in OS X). This will make all our screenshots have a similar shadow behind it.

### Use a common, neutral browser
Use an instance of **Chrome with all extensions and extras hidden**, except for the location bar.

![wdi repo](https://cloud.githubusercontent.com/assets/1327983/9769944/2a5bef4c-56fb-11e5-9ff0-646400749cc1.png)

_Figure 01. Example Screenshot_

### Scale it down to fit
Make sure the image is **no wider than 790px**, to try to keep it visible & readable in Github's style. Crop images that are taller than that down to something within that vicinity, so it doesn't totally overtake the readme. OS X's Preview is an easy way to scale images down.

### Let Github Store the Images
Rather than storing images in some folder in our lesson, **use [Github's Issue Attachment CDN](https://help.github.com/articles/issue-attachments/) to upload & link to images.** Essentially, pretend you're going to attach an image to a Github Issue, let it upload, and then copy & paste the CDN URL it gives and include that. There's a [detailed blog post here](http://solutionoptimist.com/2013/12/28/awesome-github-tricks/) to read more.

---

<a name="file-structure"></a>
## File Structure

Each lesson resource should contain the following files: README.md and slides.pdf.  Each lesson should also contain the following folders: starter-code, solution-code, and a project-related folder.  

```

- 04-statistics-fundamentals-two/

    - starter-code/
        - stats-practice.ipynb

    - solution-code/
        - stats-practice.ipynb

    - README.md


```


Some lessons will have time dedicated to assigning projects.  For these lessons, simply add a folder with the necessary componenets:

```

- 03-statistics-fundamentals/

    - project-02-exploratory-analysis/
        - starter-code/
            - general-assemblys-eda.ipynb
        - solution-code/
            - general-assemblys-eda.ipynb
        - README.md

    - starter-code/
        - stats-practice.ipynb

    - solution-code/
        - stats-practice.ipynb

    - README.md


```

### A Readme In Every Folder

**Every folder should have a readme.**

That means that looking on Github, every folder will come with some explanation of what it is & what to do with it.

### Uppercase README.md

README.md files should always be markdown, with a `.md` extension. Just cuz it's shorter.

```
$ README.md
```

### Folder Names When Including Code

**If a lesson/lab/homework needs nothing** but a readme, just include the readme & don't create any extra folders.

**If a lesson/lab/homework needs starter code**, code to get students up & running, focusing on the task at hand -  include a folder called `starter-code` next to your readme.  If students require multiple files and folders, create a subfolder within the `starter-code` that appropriately names the starter app.

```
- README.md
- starter-code/
    - app-name/
        - model-server.py
        - data.csv
        etc.
```

**If a lesson/lab/homework needs solution code**, or a completed example for students to later assess themselves, include a folder called `solution-code` next to your readme.  If the solution requires multiple files and folders, create a subfolder within the `solution-code` that appropriately names the solution app.

```
- README.md
- starter-code/
  - app-name/
        - model-server.py
        - data.csv
        etc.
- solution-code/
    - app-name/
        - model-server.py
        - data.csv
        etc.
```


<a name="material-metadata"></a>
## Material Metadata

Each lesson should include some simple metadata at the top. This should be YAML frontmatter, which will let us use it like structured data later on.

### YAML Frontmatter
We want to have as minimal information as necessary, but stuff that's still useful. And we like the idea of giving credit to the original creator of the lesson or lab.

We also want to include the type (lesson, lab) and any core competency categories this lesson covers. This will help us organize later.

```
yaml
title: 'What is Data Science?'
duration: "3:00"
creator:
    name: Jane Doe
    city: NYC
```

### Lesson Timings

It's beneficial to have suggested timing in each lesson, it gives us an idea how long we might want to take on each section.

**Timing estimates should go in increments of 5 minutes.** That's small enough to have tiny pieces of a lesson, but still evenly calculable over the whole duration. Anything more specific will be too granular.

The timing should be included in each section heading. We'll talk about how to choose which level headings in the next section (ba dum chhhh). **But timings are only necessary on H2s.** Here's an example:

```
markdown
## Loading Data into Pandas - Intro (10 mins.)
```

---

<a name="headings--text"></a>
## Headings & Text

Check out the example `template-lesson-readme.md` & `template-project-readme.md` to see an empty template you can use as a starting point for creating new materials.

Here's some codified explanation of it.

### H1 for Lesson Title

There should be only one, so it should be the first thing in the markdown, and be an H1.

It should also match exactly what's in the Trello planning doc.

### H3 & UL for Learning Objectives

Learning Objectives are good practice for us make sure students are clear about what we're trying to teach them, and we stay on track with only the most important pieces of a topic. For the record, you should write these learning objectives on the wall when you go to teach this lesson.

The actual objectives should just be an unordered list.

### H3 & UL for Preparation

This is a section to help both students & instructors know what a student needs to know before they'll understand this lesson. These could be references to specific existing lessons in the curriculum, or they could just be concepts students need to know.

It might be useful to use this list as check-for-understanding questions before you dive too deep into new materials.

### H2 for Lesson content section titles

The actual content of the lesson is the most important thing, and using H2s gives us nice visual hierarchy for breaking our lesson up into sections.

Use as many sections as you see fit, and try to give them titles that give some hint of what we're about to talk about. They don't have to be catchy or witty (though you're welcome to make them both), but they should be _clear_.

### Break up paragraphs for easy reading

We want to make these documents scannable & easily readable. Every time you're talking about a new idea or concept, or you realize the sentence you're writing doesn't add on to the previous one – make it a new paragraph.

Err on the side of having too many paragraphs if possible, even if a paragraph is only one or two sentences. It'll be easier to read.

### Use triple-tick code blocks for highlighting on Github

Use triple-tick code blocks with languages to make code auto-syntax-highlighted.

    ```
    python
      def sayHi:
        print("hi!")
    ```

You can read more about it [here](https://help.github.com/articles/github-flavored-markdown/#syntax-highlighting) if you're unfamiliar.

### Test your markdown with Github Flavored Markdown

Test your code using the [Github markdown API endpoint](https://developer.github.com/v3/markdown/#render-an-arbitrary-markdown-document), or a tool that uses it, to make sure it's working. Good options are:

- [Markdown Preview for Sublime](https://packagecontrol.io/packages/Markdown%20Preview)
- [moo.js](https://www.npmjs.com/package/moo.js)
- [Marko](https://itunes.apple.com/us/app/marko/id607997198?mt=12)
- [Marked 2](http://marked2app.com/)

<a name="contributing-materials"></a>
## Contributing Materials

See the [contributing guidelines](../contributing.md) for information on how to help!
