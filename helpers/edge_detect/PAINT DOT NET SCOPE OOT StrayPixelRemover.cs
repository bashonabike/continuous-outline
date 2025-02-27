#region Copyright
/*
 * ExamplePropertyBasedFileType file type
 * Copyright (C) 2013 ComSquare AG, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published load
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#endregion

using System.Collections.Generic;
using System.Drawing;

using PaintDotNet;
using PaintDotNet.IndirectUI;
using PaintDotNet.Effects;
using PaintDotNet.PropertySystem;

namespace StrayPixelRemover
{
    public sealed class StrayPixelsEffect
        : PropertyBasedEffect
    {
        // ----------------------------------------------------------------------
        /// <summary>
        /// Defines a user friendly name used for menu and dialog caption
        /// </summary>
        private const string StaticName = "Stray pixels remover";

        // ----------------------------------------------------------------------
        /// <summary>
        /// Defines an image used for for menu and dialog caption (may be null)
        /// </summary>
        private static Bitmap StaticImage => Resources.EffectIcon;

        // ----------------------------------------------------------------------
        /// <summary>
        /// Defines the submenu name where the effect should be placed.
        /// Prefered is one of the SubmenuNames constants (i.e. SubmenuNames.Render)
        /// (may be null)
        /// </summary>
        private const string StaticSubMenuName = "Object";

        // ----------------------------------------------------------------------
        /// <summary>
        /// Constructs an ExamplePropertyBasedEffect instance
        /// </summary>
        public StrayPixelsEffect()
            : base(StaticName, StaticImage, StaticSubMenuName, EffectFlags.Configurable | EffectFlags.SingleThreaded)
        {
        }


        /// <summary>
        /// Identifiers of the properties used by the effect
        /// </summary>
        private enum PropertyNames
        {
            ThresholdSlider,
            AlphaSlider
        }



        // Settings of the properties
        private int propThresHoldSlider;
        private int propAlphaSlider;


        // ----------------------------------------------------------------------
        /// <summary>
        /// Configure the properties of the effect.
        /// This just creates the properties not the controls used in the dialog.
        /// These properties are defining the content of the EffectToken.
        /// </summary>
        protected override PropertyCollection OnCreatePropertyCollection()
        {
            // Add properties of all types and control types (always the variant with minimal parameters)
            List<Property> props = new List<Property>
            {
                new Int32Property(PropertyNames.ThresholdSlider,2,1,9),
                new Int32Property(PropertyNames.AlphaSlider,0,0,255)
            };

            return new PropertyCollection(props);
        } /* OnCreatePropertyCollection */

        // ----------------------------------------------------------------------
        /// <summary>
        /// Configure the user interface of the effect.
        /// You may change the default control type of your properties or
        /// modify/suppress the default texts in the controls.
        /// </summary>
        protected override ControlInfo OnCreateConfigUI(PropertyCollection props)
        {
            ControlInfo configUI = CreateDefaultConfigUI(props);
            configUI.SetPropertyControlValue(PropertyNames.ThresholdSlider, ControlInfoPropertyNames.DisplayName, "Radius Threshold");
            configUI.SetPropertyControlValue(PropertyNames.AlphaSlider, ControlInfoPropertyNames.DisplayName, "Alpha Threshold");
            return configUI;
        } /* OnCreateConfigUI */

        // ----------------------------------------------------------------------
        /// <summary>
        /// Called after the token of the effect changed.
        /// This method is used to read all values of the token to instance variables.
        /// These instance variables are then used to render the surface.
        /// </summary>
        protected override void OnSetRenderInfo(PropertyBasedEffectConfigToken newToken, RenderArgs dstArgs, RenderArgs srcArgs)
        {
            // Read the current settings of the properties
            propThresHoldSlider = newToken.GetProperty<Int32Property>(PropertyNames.ThresholdSlider).Value;
            propAlphaSlider = newToken.GetProperty<Int32Property>(PropertyNames.AlphaSlider).Value;


            Selection = EnvironmentParameters.GetSelection(SrcArgs.Surface.Bounds).GetBoundsInt();

            if (SURF == null)
                SURF = new Surface(SrcArgs.Surface.Bounds.Size);
            SURF.CopySurface(SrcArgs.Surface);

            for (int y = Selection.Top; y < Selection.Bottom; y++)
            {
                if (IsCancelRequested) return;
                for (int x = Selection.Left; x < Selection.Right; x++)
                {
                    ToDelete.Clear();
                    if (TestForDelete(x, y, 0, true, true, true, true))
                    { //If we need to delete the pixel
                        SURF[x, y] = ColorBgra.Transparent; // we delete it
                        foreach (Point p in ToDelete.AsReadOnly()) //delete contiguous pixels
                        {
                            SURF[p.X, p.Y] = ColorBgra.Transparent;
                        }
                    }
                }
            }

            PdnRegion exactSelection = EnvironmentParameters.GetSelection(srcArgs.Surface.Bounds);
            DstArgs.Surface.CopySurface(SURF, exactSelection);


            base.OnSetRenderInfo(newToken, dstArgs, srcArgs);
        } /* OnSetRenderInfo */

        // ----------------------------------------------------------------------
        /// <summary>
        /// Render an area defined by a list of rectangles
        /// This function may be called multiple times to render the area of
        //  the selection on the active layer
        /// </summary>
        /// 

        Rectangle Selection;
        Surface SURF;
        readonly List<Point> ToDelete = new List<Point>();

        protected override void OnRender(Rectangle[] renderRects, int startIndex, int length)
        {
        }

        //return true if the pixel has to be deleted
        bool TestForDelete(int px, int py, int dist, bool h, bool b, bool g, bool d)
        {
            if (dist > propThresHoldSlider) return false; //too far from tested pixel
            if (px < Selection.Left || py < Selection.Top || px >= Selection.Right || py >= Selection.Bottom) return true; //out of Selection
            if (SURF[px, py].A <= propAlphaSlider) return true; // transparent
            bool ret = true;
            if (ret && h == true) ret = TestForDelete(px, py - 1, dist + 1, true, false, g, d);
            if (ret && b == true) ret = TestForDelete(px, py + 1, dist + 1, false, true, g, d);
            if (ret && g == true) ret = TestForDelete(px - 1, py, dist + 1, h, b, true, false);
            if (ret && d == true) ret = TestForDelete(px + 1, py, dist + 1, h, b, false, true);
            //if surrounded only by pixel that must be deleted and pixel that are transparent, return true.
            if (ret) ToDelete.Add(new Point(px, py));
            return ret;
        }

    }

}